// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// kimera_live_vio — long-running Kimera-VIO pipeline driven by a Unix
// domain socket.  Replaces the EuRoC watch-folder shim with true
// streaming I/O: the MCP server connects once, pushes raw grayscale
// frames + IMU samples + intrinsics, and gets the latest backend pose
// back on every request.
//
// Protocol (binary, little-endian, length-prefixed)
// -------------------------------------------------
// Header (12 bytes) before every payload:
//     uint32_t msg_type    // see below
//     uint32_t payload_len // bytes that follow this header
//     uint32_t reserved    // 0
//
// Client -> Server messages
//     1 PING                 (no payload)
//     2 INTRINSICS           u32 w, u32 h, f64 fx, f64 fy, f64 cx, f64 cy
//                            f64 k1, f64 k2, f64 p1, f64 p2
//     3 IMU                  u32 count, [u64 ts_ns, f64 gx, f64 gy, f64 gz,
//                                         f64 ax, f64 ay, f64 az] * count
//     4 FRAME                u64 ts_ns, u32 w, u32 h, w*h bytes grayscale
//     5 RESET                (no payload)
//
// Server -> Client responses (always one response per request)
//     11 PONG                (no payload)
//     12 OK                  (no payload — INTRINSICS / IMU / RESET ack)
//     13 POSE                u64 ts_ns, f64 tx, f64 ty, f64 tz,
//                            f64 qw, f64 qx, f64 qy, f64 qz,
//                            u32 state (0 ok, 1 uninitialized, 2 lost)
//     20 ERROR               u32 msg_len, msg_len bytes utf-8
//
// The binary listens on AF_UNIX SOCK_STREAM at --socket_path.  It
// accepts one connection at a time and serves it until close.  If the
// MCP server reconnects, the binary tears down the pipeline state and
// rebuilds it on the next INTRINSICS (or FRAME with defaults).

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "kimera-vio/backend/VioBackend-definitions.h"
#include "kimera-vio/frontend/Frame.h"
#include "kimera-vio/frontend/CameraParams.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/pipeline/MonoImuPipeline.h"
#include "kimera-vio/pipeline/Pipeline-definitions.h"
#include "kimera-vio/dataprovider/DataProviderInterface.h"

DEFINE_string(socket_path, "/tmp/kimera-vio.sock",
              "Path to the AF_UNIX socket to listen on.");
DEFINE_string(params_folder_path, "/opt/kimera-params/EurocMonoLive",
              "Folder containing the YAML param files (baseline; overridden "
              "in-memory by the INTRINSICS message).");

namespace VIO {

// ── Custom data provider ────────────────────────────────────────────────────

class LiveDataProvider : public DataProviderInterface {
 public:
  LiveDataProvider() = default;
  ~LiveDataProvider() override = default;

  // No background spin needed — frames and IMU are pushed from the
  // socket reader thread directly via deliverFrame() / deliverImu().
  // We just block until shutdown to keep the pipeline's
  // waitForShutdown() happy.
  bool spin() override {
    std::unique_lock<std::mutex> lk(spin_mu_);
    spin_cv_.wait(lk, [this] { return shutdown_.load(); });
    return false;
  }

  bool hasData() const override { return !shutdown_.load(); }

  void shutdown() override {
    shutdown_ = true;
    std::lock_guard<std::mutex> lk(spin_mu_);
    spin_cv_.notify_all();
  }

  // Called by the socket reader thread.
  void deliverImu(int64_t ts_ns, const std::array<double, 6>& acc_gyr) {
    if (!imu_single_callback_) return;
    ImuAccGyr v;
    // Kimera's IMU storage is [ax, ay, az, gx, gy, gz] in the
    // ImuAccGyr layout (acc first, gyr second).  Match that.
    v << acc_gyr[3], acc_gyr[4], acc_gyr[5],
         acc_gyr[0], acc_gyr[1], acc_gyr[2];
    imu_single_callback_(ImuMeasurement(static_cast<Timestamp>(ts_ns), v));
  }

  void deliverFrame(int64_t ts_ns, const CameraParams& cam_param,
                    cv::Mat&& gray) {
    if (!left_frame_callback_) return;
    auto f = std::make_unique<Frame>(next_frame_id_++,
                                     static_cast<Timestamp>(ts_ns),
                                     cam_param, gray);
    left_frame_callback_(std::move(f));
  }

 private:
  std::atomic<FrameId> next_frame_id_{0};
  std::mutex              spin_mu_;
  std::condition_variable spin_cv_;
};

// Re-expose the protected backend-output registration so we can attach
// our pose-capture callback from outside the pipeline hierarchy.  This
// is what KimeraVIO.cpp would do if it weren't all friend-based.
class LiveMonoPipeline : public MonoImuPipeline {
 public:
  using MonoImuPipeline::MonoImuPipeline;
  using MonoImuPipeline::registerBackendOutputCallback;
};

}  // namespace VIO

// ── Wire protocol ───────────────────────────────────────────────────────────

namespace {

enum MsgType : uint32_t {
  kMsgPing       = 1,
  kMsgIntrinsics = 2,
  kMsgImu        = 3,
  kMsgFrame      = 4,
  kMsgReset      = 5,

  kRspPong  = 11,
  kRspOk    = 12,
  kRspPose  = 13,
  kRspError = 20,
};

struct Header {
  uint32_t msg_type;
  uint32_t payload_len;
  uint32_t reserved;
} __attribute__((packed));

bool ReadExact(int fd, void* buf, size_t n) {
  uint8_t* p = static_cast<uint8_t*>(buf);
  size_t remaining = n;
  while (remaining > 0) {
    ssize_t got = ::read(fd, p, remaining);
    if (got == 0) return false;                     // peer closed
    if (got < 0) {
      if (errno == EINTR) continue;
      PLOG(WARNING) << "read";
      return false;
    }
    p         += got;
    remaining -= static_cast<size_t>(got);
  }
  return true;
}

bool WriteExact(int fd, const void* buf, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(buf);
  size_t remaining = n;
  while (remaining > 0) {
    ssize_t put = ::write(fd, p, remaining);
    if (put <= 0) {
      if (put < 0 && errno == EINTR) continue;
      PLOG(WARNING) << "write";
      return false;
    }
    p         += put;
    remaining -= static_cast<size_t>(put);
  }
  return true;
}

bool SendResponse(int fd, uint32_t type, const void* payload, uint32_t len) {
  Header h{type, len, 0};
  if (!WriteExact(fd, &h, sizeof(h))) return false;
  if (len > 0 && !WriteExact(fd, payload, len)) return false;
  return true;
}

bool SendError(int fd, const std::string& msg) {
  uint32_t n = static_cast<uint32_t>(msg.size());
  std::vector<uint8_t> body(sizeof(uint32_t) + n);
  std::memcpy(body.data(), &n, sizeof(uint32_t));
  std::memcpy(body.data() + sizeof(uint32_t), msg.data(), n);
  return SendResponse(fd, kRspError, body.data(),
                      static_cast<uint32_t>(body.size()));
}

}  // namespace

// ── Pipeline wrapper ────────────────────────────────────────────────────────

class LivePipeline {
 public:
  explicit LivePipeline(const std::string& params_folder)
      : params_folder_(params_folder) {}

  ~LivePipeline() { shutdown(); }

  // Apply external intrinsic overrides (called before rebuild()).
  void setIntrinsics(int w, int h, double fx, double fy,
                     double cx, double cy,
                     double k1, double k2, double p1, double p2) {
    std::lock_guard<std::mutex> lk(intr_mu_);
    width_  = w;
    height_ = h;
    fx_     = fx;
    fy_     = fy;
    cx_     = cx;
    cy_     = cy;
    k_      = {k1, k2, p1, p2};
    intrinsics_set_ = true;
  }

  // (Re)build pipeline.  If one is already running, shut it down first.
  // Returns false if construction failed.
  bool rebuild() {
    shutdown();

    LOG(INFO) << "[kimera-live] (re)building pipeline; params=" << params_folder_;
    vio_params_ = std::make_unique<VIO::VioParams>(params_folder_);

    // In-memory intrinsic override — only the left camera matters for
    // MonoImu.  We keep the right-camera entry present (Kimera expects
    // it in EuRoC params even in mono mode) but it's unused.
    if (intrinsics_set_ && !vio_params_->camera_params_.empty()) {
      std::lock_guard<std::mutex> lk(intr_mu_);
      auto& cp = vio_params_->camera_params_[0];
      cp.intrinsics_ = {fx_, fy_, cx_, cy_};
      cp.image_size_ = cv::Size(width_, height_);
      cp.K_          = (cv::Mat_<double>(3, 3) <<
                          fx_, 0.0, cx_,
                          0.0, fy_, cy_,
                          0.0, 0.0, 1.0);
      cp.distortion_coeff_ = {k_[0], k_[1], k_[2], k_[3]};
      cv::Mat dmat(1, 4, CV_64F, cp.distortion_coeff_.data());
      cp.distortion_coeff_mat_ = dmat.clone();
    }
    cam_param_ = vio_params_->camera_params_.empty()
                     ? VIO::CameraParams{}
                     : vio_params_->camera_params_[0];

    data_provider_ = std::make_shared<VIO::LiveDataProvider>();
    pipeline_      = std::make_unique<VIO::LiveMonoPipeline>(*vio_params_);

    // Backend output callback — capture the latest VIO pose under a mutex.
    pipeline_->registerBackendOutputCallback(
        [this](const VIO::BackendOutput::ConstPtr& out) {
          if (!out) return;
          const auto& pose = out->W_State_Blkf_.pose_;
          const auto& t    = pose.translation();
          const auto  q    = pose.rotation().toQuaternion();
          std::lock_guard<std::mutex> lk(latest_mu_);
          latest_have_pose_ = true;
          latest_ts_ns_     = out->timestamp_;
          latest_tx_        = t.x();
          latest_ty_        = t.y();
          latest_tz_        = t.z();
          latest_qw_        = q.w();
          latest_qx_        = q.x();
          latest_qy_        = q.y();
          latest_qz_        = q.z();
        });

    pipeline_->registerShutdownCallback(
        std::bind(&VIO::DataProviderInterface::shutdown, data_provider_));

    // Wire data provider -> pipeline input queues.
    data_provider_->registerImuSingleCallback(std::bind(
        &VIO::Pipeline::fillSingleImuQueue, pipeline_.get(),
        std::placeholders::_1));
    data_provider_->registerLeftFrameCallback(std::bind(
        &VIO::Pipeline::fillLeftFrameQueue, pipeline_.get(),
        std::placeholders::_1));

    // Launch pipeline threads + a thread to spin the (no-op) data provider
    // so the pipeline thinks data is flowing.
    provider_thread_ = std::thread([dp = data_provider_]() {
      dp->spin();
    });
    pipeline_thread_ = std::thread([this]() {
      pipeline_->spin();
    });
    return true;
  }

  void deliverImu(int64_t ts_ns, const std::array<double, 6>& acc_gyr) {
    if (!data_provider_) return;
    data_provider_->deliverImu(ts_ns, acc_gyr);
  }

  void deliverFrame(int64_t ts_ns, cv::Mat&& gray) {
    if (!data_provider_) return;
    data_provider_->deliverFrame(ts_ns, cam_param_, std::move(gray));
  }

  struct LatestPose {
    bool     have_pose;
    int64_t  ts_ns;
    double   tx, ty, tz;
    double   qw, qx, qy, qz;
  };

  LatestPose latest() {
    std::lock_guard<std::mutex> lk(latest_mu_);
    return {latest_have_pose_,
            latest_ts_ns_,
            latest_tx_, latest_ty_, latest_tz_,
            latest_qw_, latest_qx_, latest_qy_, latest_qz_};
  }

  size_t framesDelivered() const { return frames_delivered_; }
  void   bumpFrameCount()       { frames_delivered_++; }

  void shutdown() {
    if (!pipeline_) return;
    LOG(INFO) << "[kimera-live] shutting down current pipeline";
    if (data_provider_) data_provider_->shutdown();
    pipeline_->shutdown();
    if (provider_thread_.joinable()) provider_thread_.join();
    if (pipeline_thread_.joinable()) pipeline_thread_.join();
    pipeline_.reset();
    data_provider_.reset();
    vio_params_.reset();
    std::lock_guard<std::mutex> lk(latest_mu_);
    latest_have_pose_ = false;
    frames_delivered_ = 0;
  }

  bool isBuilt() const { return pipeline_ != nullptr; }

 private:
  std::string params_folder_;

  // Intrinsic overrides — accumulated independently of pipeline lifecycle.
  std::mutex intr_mu_;
  bool   intrinsics_set_ = false;
  int    width_ = 0, height_ = 0;
  double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  std::array<double, 4> k_ = {0, 0, 0, 0};

  std::unique_ptr<VIO::VioParams>        vio_params_;
  std::shared_ptr<VIO::LiveDataProvider> data_provider_;
  std::unique_ptr<VIO::LiveMonoPipeline> pipeline_;
  std::thread                            provider_thread_;
  std::thread                            pipeline_thread_;

  // Per-Frame camera params (set from vio_params_ at rebuild time).
  VIO::CameraParams cam_param_;

  std::mutex latest_mu_;
  bool       latest_have_pose_ = false;
  int64_t    latest_ts_ns_     = 0;
  double     latest_tx_=0, latest_ty_=0, latest_tz_=0;
  double     latest_qw_=1, latest_qx_=0, latest_qy_=0, latest_qz_=0;

  std::atomic<size_t> frames_delivered_{0};
};

// ── Per-connection handler ──────────────────────────────────────────────────

static void HandleConnection(int conn_fd, LivePipeline& pipeline) {
  LOG(INFO) << "[kimera-live] client connected on fd=" << conn_fd;
  pipeline.shutdown();   // ensure a fresh state per connection

  while (true) {
    Header hdr{};
    if (!ReadExact(conn_fd, &hdr, sizeof(hdr))) break;

    std::vector<uint8_t> payload;
    if (hdr.payload_len > 0) {
      if (hdr.payload_len > (64u << 20)) {  // 64 MiB hard cap per message
        SendError(conn_fd, "payload too large");
        break;
      }
      payload.resize(hdr.payload_len);
      if (!ReadExact(conn_fd, payload.data(), payload.size())) break;
    }

    switch (hdr.msg_type) {
      case kMsgPing:
        SendResponse(conn_fd, kRspPong, nullptr, 0);
        break;

      case kMsgIntrinsics: {
        // w(u32) + h(u32) + 8 doubles (fx,fy,cx,cy,k1,k2,p1,p2)
        if (payload.size() != 4 + 4 + 8 * 8) {
          SendError(conn_fd, "bad INTRINSICS payload size");
          break;
        }
        const uint8_t* p = payload.data();
        uint32_t w, h;
        std::memcpy(&w, p,     4); p += 4;
        std::memcpy(&h, p,     4); p += 4;
        double fx, fy, cx, cy, k1, k2, p1, p2;
        auto rd = [&p](double& d) { std::memcpy(&d, p, 8); p += 8; };
        rd(fx); rd(fy); rd(cx); rd(cy);
        rd(k1); rd(k2); rd(p1); rd(p2);
        pipeline.setIntrinsics(static_cast<int>(w), static_cast<int>(h),
                               fx, fy, cx, cy, k1, k2, p1, p2);
        if (!pipeline.rebuild()) {
          SendError(conn_fd, "pipeline rebuild failed");
        } else {
          SendResponse(conn_fd, kRspOk, nullptr, 0);
        }
        break;
      }

      case kMsgImu: {
        if (payload.size() < 4) {
          SendError(conn_fd, "bad IMU payload (too short)");
          break;
        }
        uint32_t count;
        std::memcpy(&count, payload.data(), 4);
        const size_t per = 8 + 6 * 8;  // ts_ns + 6 doubles
        if (payload.size() != 4 + per * count) {
          SendError(conn_fd, "bad IMU payload size");
          break;
        }
        if (!pipeline.isBuilt()) {
          // Buffer-free fallback: silently drop until we have a pipeline.
          // Apps should send INTRINSICS first.
          SendResponse(conn_fd, kRspOk, nullptr, 0);
          break;
        }
        const uint8_t* p = payload.data() + 4;
        for (uint32_t i = 0; i < count; ++i) {
          uint64_t ts_ns;
          std::memcpy(&ts_ns, p, 8); p += 8;
          std::array<double, 6> v;
          for (auto& x : v) { std::memcpy(&x, p, 8); p += 8; }
          pipeline.deliverImu(static_cast<int64_t>(ts_ns), v);
        }
        SendResponse(conn_fd, kRspOk, nullptr, 0);
        break;
      }

      case kMsgFrame: {
        if (payload.size() < 8 + 4 + 4) {
          SendError(conn_fd, "bad FRAME header");
          break;
        }
        uint64_t ts_ns;
        uint32_t w, h;
        std::memcpy(&ts_ns, payload.data(),     8);
        std::memcpy(&w,     payload.data() + 8, 4);
        std::memcpy(&h,     payload.data() + 12, 4);
        const size_t expected = 16 + static_cast<size_t>(w) * h;
        if (payload.size() != expected) {
          SendError(conn_fd, "bad FRAME payload size");
          break;
        }
        if (!pipeline.isBuilt()) {
          // No intrinsics yet — bring up the pipeline with the defaults
          // baked into params_folder so the connection still functions.
          if (!pipeline.rebuild()) {
            SendError(conn_fd, "pipeline rebuild failed (no intrinsics)");
            break;
          }
        }
        cv::Mat gray(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
        std::memcpy(gray.data, payload.data() + 16,
                    static_cast<size_t>(w) * h);
        pipeline.deliverFrame(static_cast<int64_t>(ts_ns), std::move(gray));
        pipeline.bumpFrameCount();

        // Reply with the latest pose available so far.
        auto lp = pipeline.latest();
        uint8_t  buf[8 + 8 * 7 + 4];
        uint8_t* p = buf;
        uint64_t ts = static_cast<uint64_t>(lp.ts_ns);
        std::memcpy(p, &ts, 8); p += 8;
        auto wr = [&p](double d) { std::memcpy(p, &d, 8); p += 8; };
        wr(lp.tx); wr(lp.ty); wr(lp.tz);
        wr(lp.qw); wr(lp.qx); wr(lp.qy); wr(lp.qz);
        uint32_t state;
        if (lp.have_pose)                  state = 0;
        else if (pipeline.framesDelivered() < 32) state = 1;  // uninit
        else                                state = 2;        // lost
        std::memcpy(p, &state, 4); p += 4;
        SendResponse(conn_fd, kRspPose, buf, sizeof(buf));
        break;
      }

      case kMsgReset:
        pipeline.shutdown();
        SendResponse(conn_fd, kRspOk, nullptr, 0);
        break;

      default:
        SendError(conn_fd, "unknown msg_type");
        break;
    }
  }

  LOG(INFO) << "[kimera-live] client disconnected (fd=" << conn_fd << ")";
  ::close(conn_fd);
}

// ── Main loop ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  // Kimera defaults `--visualize=true`, which pulls in VTK/OpenGL and
  // crashes the moment the pipeline tries to open an X display — we
  // run headless in docker.  Force-disable before flag parsing so any
  // operator override on the command line still wins.
  FLAGS_visualize  = false;
  FLAGS_use_lcd    = false;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;

  // Bind AF_UNIX socket.
  ::unlink(FLAGS_socket_path.c_str());  // best-effort cleanup
  int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd < 0) { PLOG(FATAL) << "socket"; }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, FLAGS_socket_path.c_str(),
               sizeof(addr.sun_path) - 1);
  if (::bind(listen_fd,
             reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    PLOG(FATAL) << "bind " << FLAGS_socket_path;
  }
  if (::chmod(FLAGS_socket_path.c_str(), 0666) < 0) {
    PLOG(WARNING) << "chmod " << FLAGS_socket_path;
  }
  if (::listen(listen_fd, 4) < 0) { PLOG(FATAL) << "listen"; }

  LOG(INFO) << "[kimera-live] listening on " << FLAGS_socket_path;
  LivePipeline pipeline(FLAGS_params_folder_path);

  // Single-connection-at-a-time server.  The MCP server is the sole
  // expected client; if it reconnects we tear down state and start
  // fresh in HandleConnection().
  while (true) {
    int conn_fd = ::accept(listen_fd, nullptr, nullptr);
    if (conn_fd < 0) {
      if (errno == EINTR) continue;
      PLOG(WARNING) << "accept";
      continue;
    }
    HandleConnection(conn_fd, pipeline);
  }
  return 0;
}
