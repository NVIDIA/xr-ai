package com.nvidia.xrai.streamkitsample

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.nvidia.xrai.streamkitsample.streamkit.ConnectionState
import com.nvidia.xrai.streamkitsample.streamkit.config.AudioConfig
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ── Color tokens (match web client's CSS variables) ───────────────────────────

private val ColorGreen    = Color(0xFF34C759)
private val ColorOrange   = Color(0xFFFF9500)
private val ColorRed      = Color(0xFFFF3B30)
private val ColorBlue     = Color(0xFF007AFF)
private val ColorSecondary = Color(0x993C3C43)   // 60 % opacity gray
private val ColorSeparator = Color(0x1F3C3C43)   // 12 % opacity gray
private val ColorCardBg   = Color(0xFFFFFFFF)
private val ColorPageBg   = Color(0xFFF2F2F7)

// ─────────────────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            StreamKitTheme {
                StreamKitSampleApp()
            }
        }
    }
}

// ── App-level theme ────────────────────────────────────────────────────────────

@Composable
private fun StreamKitTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}

// ── Root screen ────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StreamKitSampleApp(vm: AppViewModel = viewModel()) {
    val snackbarHostState = remember { SnackbarHostState() }

    // Show errors as snackbars (mirrors the web toast / iOS alert).
    LaunchedEffect(vm.lastError) {
        val err = vm.lastError ?: return@LaunchedEffect
        snackbarHostState.showSnackbar(
            message = err,
            duration = SnackbarDuration.Long,
        )
        vm.clearError()
    }

    Scaffold(
        containerColor = ColorPageBg,
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = { Text("StreamKit Sample", style = MaterialTheme.typography.titleLarge) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = ColorPageBg,
                ),
            )
        },
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Spacer(Modifier.height(4.dp))
            ConnectionSection(vm)
            MediaSection(vm)
            AgentSection(vm)
            DataChannelSection(vm)
            if (vm.receivedMessages.isNotEmpty()) {
                ReceivedSection(vm)
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}

// ── Section scaffold ──────────────────────────────────────────────────────────

@Composable
private fun SectionCard(
    title: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit,
) {
    Column(modifier = modifier) {
        Text(
            text = title.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(
                letterSpacing = 0.06.sp,
                color = ColorSecondary,
            ),
            modifier = Modifier.padding(start = 4.dp, bottom = 6.dp),
        )
        Card(
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = ColorCardBg),
            elevation = CardDefaults.cardElevation(0.dp),
        ) {
            content()
        }
    }
}

@Composable
private fun CardRow(
    modifier: Modifier = Modifier,
    showDivider: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    Column {
        Row(
            modifier = modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 11.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content,
        )
        if (showDivider) HorizontalDivider(color = ColorSeparator, thickness = 0.5.dp)
    }
}

// ── Connection state badge ─────────────────────────────────────────────────────

@Composable
private fun ConnectionStateDot(state: ConnectionState) {
    val dotColor = when (state) {
        ConnectionState.DISCONNECTED -> ColorSecondary
        ConnectionState.CONNECTING   -> ColorOrange
        ConnectionState.CONNECTED    -> ColorGreen
        ConnectionState.RECONNECTING -> ColorOrange
    }
    val isPulsing = state == ConnectionState.CONNECTING || state == ConnectionState.RECONNECTING

    if (isPulsing) {
        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
        val alpha by infiniteTransition.animateFloat(
            initialValue = 1f,
            targetValue = 0.3f,
            animationSpec = infiniteRepeatable(
                animation = tween(600, easing = LinearEasing),
                repeatMode = RepeatMode.Reverse,
            ),
            label = "dot-alpha",
        )
        Box(
            modifier = Modifier
                .size(8.dp)
                .alpha(alpha)
                .background(color = dotColor, shape = CircleShape),
        )
    } else {
        Box(
            modifier = Modifier
                .size(8.dp)
                .background(color = dotColor, shape = CircleShape),
        )
    }
}

// ── Connection section ────────────────────────────────────────────────────────

@Composable
private fun ConnectionSection(vm: AppViewModel) {
    val state = vm.connectionState
    val isDisconnected = state == ConnectionState.DISCONNECTED
    val isTransitioning = state == ConnectionState.CONNECTING || state == ConnectionState.RECONNECTING

    SectionCard(title = "Connection") {
        // State row
        CardRow {
            Text("State", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))
            Row(verticalAlignment = Alignment.CenterVertically) {
                ConnectionStateDot(state)
                Spacer(Modifier.width(6.dp))
                val label = when (state) {
                    ConnectionState.DISCONNECTED -> "Disconnected"
                    ConnectionState.CONNECTING   -> "Connecting…"
                    ConnectionState.CONNECTED    -> "Connected"
                    ConnectionState.RECONNECTING -> "Reconnecting…"
                }
                Text(label, style = MaterialTheme.typography.bodySmall, color = ColorSecondary)
            }
        }

        // Config fields — dimmed while connected / connecting
        val fieldAlpha = if (isDisconnected) 1f else 0.4f

        FieldRow(
            label = "Host / IP",
            value = vm.host,
            onValueChange = { vm.host = it },
            enabled = isDisconnected,
            alpha = fieldAlpha,
            keyboardType = KeyboardType.Uri,
            placeholder = "192.168.1.100",
        )
        FieldRow(
            label = "Port",
            value = vm.port,
            onValueChange = { vm.port = it },
            enabled = isDisconnected,
            alpha = fieldAlpha,
            keyboardType = KeyboardType.Number,
            placeholder = "7880",
        )
        // HTTPS toggle for the default token endpoint. The LiveKit signaling
        // socket on 7880 is always plain ws:// in the xr-ai reference
        // deployment, so this only flips http:// ↔ https:// for the token URL.
        CardRow {
            Box(modifier = Modifier.alpha(fieldAlpha)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "HTTPS token",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.weight(1f))
                    Switch(
                        checked = vm.secure,
                        onCheckedChange = { vm.secure = it },
                        enabled = isDisconnected,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color.White,
                            checkedTrackColor = ColorGreen,
                        ),
                    )
                }
            }
        }
        FieldRow(
            label = "Token",
            value = vm.tokenInput,
            onValueChange = { vm.tokenInput = it },
            enabled = isDisconnected,
            alpha = fieldAlpha,
            placeholder = "Paste JWT directly",
            visualTransformation = PasswordVisualTransformation(),
        )
        FieldRow(
            label = "Token URL",
            value = vm.tokenServerURL,
            onValueChange = { vm.tokenServerURL = it },
            enabled = isDisconnected,
            alpha = fieldAlpha,
            keyboardType = KeyboardType.Uri,
            placeholder = "/token (default)",
        )
        FieldRow(
            label = "Identity",
            value = vm.identity,
            onValueChange = { vm.identity = it },
            enabled = isDisconnected,
            alpha = fieldAlpha,
            placeholder = "android-client",
            showDivider = true,
        )

        // Connect / Disconnect button
        CardRow(showDivider = false) {
            val (label, containerColor) = when {
                isDisconnected    -> "Connect" to ColorBlue
                isTransitioning   -> (if (state == ConnectionState.CONNECTING) "Connecting…" else "Reconnecting…") to ColorSecondary
                else              -> "Disconnect" to ColorRed
            }
            Button(
                onClick = {
                    if (isDisconnected) vm.connect() else vm.disconnect()
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isTransitioning,
                colors = ButtonDefaults.buttonColors(containerColor = containerColor),
                shape = RoundedCornerShape(8.dp),
            ) {
                Text(label)
            }
        }
    }
}

@Composable
private fun FieldRow(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    enabled: Boolean,
    alpha: Float,
    placeholder: String = "",
    keyboardType: KeyboardType = KeyboardType.Text,
    visualTransformation: VisualTransformation = VisualTransformation.None,
    showDivider: Boolean = true,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .alpha(alpha)
            .padding(horizontal = 16.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.width(90.dp),
        )
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            enabled = enabled,
            placeholder = {
                Text(
                    placeholder,
                    style = MaterialTheme.typography.bodyMedium,
                    color = ColorSecondary,
                )
            },
            singleLine = true,
            visualTransformation = visualTransformation,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color.Transparent,
                unfocusedBorderColor = Color.Transparent,
                disabledBorderColor = Color.Transparent,
                focusedContainerColor = Color.Transparent,
                unfocusedContainerColor = Color.Transparent,
                disabledContainerColor = Color.Transparent,
            ),
            textStyle = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f),
        )
    }
    if (showDivider) HorizontalDivider(color = ColorSeparator, thickness = 0.5.dp)
}

// ── Media section ─────────────────────────────────────────────────────────────

@Composable
private fun MediaSection(vm: AppViewModel) {
    val isConnected = vm.connectionState == ConnectionState.CONNECTED
    val context = LocalContext.current

    // Permission launchers
    val micPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startAudio()
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startCamera()
    }

    SectionCard(title = "Media") {
        // Audio mode dropdown — disabled while mic is active
        AudioModeRow(
            mode = vm.audioMode,
            onModeChange = { vm.audioMode = it },
            enabled = !vm.isAudioActive,
        )

        // Microphone toggle
        CardRow {
            val audioLabel = if (vm.isAudioActive) "Stop Microphone" else "Start Microphone"
            val audioColor = if (vm.isAudioActive) ColorRed else ColorSecondary
            Button(
                onClick = {
                    if (vm.isAudioActive) {
                        vm.stopAudio()
                    } else {
                        val hasPerm = ContextCompat.checkSelfPermission(
                            context, Manifest.permission.RECORD_AUDIO
                        ) == PackageManager.PERMISSION_GRANTED
                        if (hasPerm) vm.startAudio()
                        else micPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = isConnected,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isConnected) audioColor else ColorSecondary,
                ),
                shape = RoundedCornerShape(8.dp),
            ) { Text(audioLabel) }
        }

        // Microphone status
        CardRow {
            Text("Microphone", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))
            val (statusText, statusColor) = when {
                vm.isAudioActive -> "Live" to ColorGreen
                isConnected      -> "Idle" to ColorSecondary
                else             -> "Not connected" to ColorSecondary
            }
            Text(statusText, style = MaterialTheme.typography.bodyMedium, color = statusColor)
        }

        // Camera selector — auto-populated from CameraManager. Hidden while
        // camera is active and when the device exposes no cameras.
        if (!vm.isCameraActive && vm.availableCameras.isNotEmpty()) {
            CameraSelectorRow(
                cameras = vm.availableCameras,
                selectedId = vm.selectedCameraId,
                onSelect = { vm.selectedCameraId = it },
                enabled = isConnected,
            )
        }

        // Camera toggle
        CardRow {
            val camLabel = if (vm.isCameraActive) "Stop Camera" else "Start Camera"
            val camColor = if (vm.isCameraActive) ColorRed else ColorSecondary
            Button(
                onClick = {
                    if (vm.isCameraActive) {
                        vm.stopCamera()
                    } else {
                        val hasPerm = ContextCompat.checkSelfPermission(
                            context, Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                        if (hasPerm) vm.startCamera()
                        else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = isConnected,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isConnected) camColor else ColorSecondary,
                ),
                shape = RoundedCornerShape(8.dp),
            ) { Text(camLabel) }
        }

        // Camera status
        CardRow {
            Text("Camera", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))
            val (statusText, statusColor) = when {
                vm.isCameraActive -> "Streaming" to ColorGreen
                isConnected       -> "Idle" to ColorSecondary
                else              -> "Not connected" to ColorSecondary
            }
            Text(statusText, style = MaterialTheme.typography.bodyMedium, color = statusColor)
        }

        // Camera on demand toggle — always visible so the user can set the
        // preference before connecting.
        CardRow(showDivider = false) {
            Text("On demand", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))
            Switch(
                checked = vm.cameraOnDemand,
                onCheckedChange = { vm.cameraOnDemand = it },
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = ColorGreen,
                ),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AudioModeRow(
    mode: AudioConfig.MicrophoneMode,
    onModeChange: (AudioConfig.MicrophoneMode) -> Unit,
    enabled: Boolean,
) {
    var expanded by remember { mutableStateOf(false) }
    val options = listOf(
        AudioConfig.MicrophoneMode.VOICE_PROCESSING to "Voice Processing",
        AudioConfig.MicrophoneMode.SOFTWARE_PROCESSING to "Software (AEC on)",
        AudioConfig.MicrophoneMode.RAW to "Raw (no DSP)",
    )
    val selectedLabel = options.firstOrNull { it.first == mode }?.second ?: "—"

    CardRow {
        Text(
            "Mic Mode",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.width(90.dp),
        )
        Spacer(Modifier.weight(1f))
        ExposedDropdownMenuBox(
            expanded = expanded && enabled,
            onExpandedChange = { if (enabled) expanded = !expanded },
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable),
            ) {
                Text(
                    selectedLabel,
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (enabled) MaterialTheme.colorScheme.onSurface else ColorSecondary,
                )
                ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded && enabled)
            }
            ExposedDropdownMenu(
                expanded = expanded && enabled,
                onDismissRequest = { expanded = false },
            ) {
                options.forEach { (modeOption, label) ->
                    DropdownMenuItem(
                        text = { Text(label) },
                        onClick = {
                            onModeChange(modeOption)
                            expanded = false
                        },
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CameraSelectorRow(
    cameras: List<CameraInfo>,
    selectedId: String?,
    onSelect: (String) -> Unit,
    enabled: Boolean,
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedLabel = cameras.firstOrNull { it.id == selectedId }?.displayName ?: "—"

    CardRow {
        Text(
            "Camera",
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.width(90.dp),
        )
        Spacer(Modifier.weight(1f))
        ExposedDropdownMenuBox(
            expanded = expanded && enabled,
            onExpandedChange = { if (enabled) expanded = !expanded },
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable),
            ) {
                Text(
                    selectedLabel,
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (enabled) MaterialTheme.colorScheme.onSurface else ColorSecondary,
                )
                ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded && enabled)
            }
            ExposedDropdownMenu(
                expanded = expanded && enabled,
                onDismissRequest = { expanded = false },
            ) {
                cameras.forEach { camera ->
                    DropdownMenuItem(
                        text = { Text(camera.displayName) },
                        onClick = {
                            onSelect(camera.id)
                            expanded = false
                        },
                    )
                }
            }
        }
    }
}

// ── Agent section ─────────────────────────────────────────────────────────────

@Composable
private fun AgentSection(vm: AppViewModel) {
    val isConnected = vm.connectionState == ConnectionState.CONNECTED

    SectionCard(title = "Agent") {
        CardRow(showDivider = false) {
            Text("Status", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.weight(1f))

            val dotColor = when {
                !isConnected                   -> ColorSecondary
                vm.agentStatus == "processing" -> ColorOrange
                vm.agentStatus == "idle"       -> ColorGreen
                else                           -> ColorSecondary
            }
            val isPulsing = isConnected && vm.agentStatus == "processing"

            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isPulsing) {
                    val infiniteTransition = rememberInfiniteTransition(label = "agent-pulse")
                    val alpha by infiniteTransition.animateFloat(
                        initialValue = 1f,
                        targetValue = 0.3f,
                        animationSpec = infiniteRepeatable(
                            animation = tween(700, easing = LinearEasing),
                            repeatMode = RepeatMode.Reverse,
                        ),
                        label = "agent-dot-alpha",
                    )
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .alpha(alpha)
                            .background(color = dotColor, shape = CircleShape),
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .background(color = dotColor, shape = CircleShape),
                    )
                }
                Spacer(Modifier.width(6.dp))
                val agentLabel = when {
                    !isConnected                   -> "—"
                    vm.agentStatus == "idle"       -> "Idle"
                    vm.agentStatus == "processing" -> "Processing…"
                    else                           -> "Unknown"
                }
                Text(agentLabel, style = MaterialTheme.typography.bodySmall, color = ColorSecondary)
            }
        }
    }
}

// ── Data channel section ──────────────────────────────────────────────────────

@Composable
private fun DataChannelSection(vm: AppViewModel) {
    val isConnected = vm.connectionState == ConnectionState.CONNECTED
    var messageText by remember { mutableStateOf("") }

    SectionCard(title = "Data Channel") {
        // Ping button
        CardRow {
            Button(
                onClick = { vm.sendPing() },
                modifier = Modifier.fillMaxWidth(),
                enabled = isConnected,
                colors = ButtonDefaults.buttonColors(containerColor = ColorSecondary),
                shape = RoundedCornerShape(8.dp),
            ) { Text("Send Ping") }
        }

        // Custom message input + send button
        CardRow(showDivider = false) {
            OutlinedTextField(
                value = messageText,
                onValueChange = { messageText = it },
                placeholder = {
                    Text(
                        "Custom message…",
                        style = MaterialTheme.typography.bodyMedium,
                        color = ColorSecondary,
                    )
                },
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color.Transparent,
                    unfocusedBorderColor = Color.Transparent,
                    focusedContainerColor = Color.Transparent,
                    unfocusedContainerColor = Color.Transparent,
                ),
                textStyle = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(1f),
            )
            Spacer(Modifier.width(8.dp))
            Button(
                onClick = {
                    vm.sendCustom(messageText)
                    messageText = ""
                },
                enabled = isConnected && messageText.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = ColorBlue),
                shape = RoundedCornerShape(8.dp),
            ) { Text("Send") }
        }
    }
}

// ── Received messages section ─────────────────────────────────────────────────

@Composable
private fun ReceivedSection(vm: AppViewModel) {
    val timeFormat = remember { SimpleDateFormat("HH:mm:ss", Locale.getDefault()) }

    SectionCard(title = "Received") {
        Column {
            vm.receivedMessages.forEachIndexed { index, msg ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.Top,
                ) {
                    Text(
                        text = msg.text,
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                            fontSize = 12.sp,
                        ),
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(Modifier.width(12.dp))
                    Text(
                        text = timeFormat.format(Date(msg.timestamp)),
                        style = MaterialTheme.typography.labelSmall,
                        color = ColorSecondary,
                    )
                }
                if (index < vm.receivedMessages.size - 1) {
                    HorizontalDivider(color = ColorSeparator, thickness = 0.5.dp)
                }
            }
        }
    }
}
