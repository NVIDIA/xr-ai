# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import unittest
from pathlib import Path


class UITests(unittest.TestCase):
    def test_gpu_memory_unavailable_does_not_render_zero_capacity(self):
        page = Path(__file__).resolve().parents[2] / "deployments/brev/ui.html"
        result = subprocess.run(["node", "-e", """
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const elements=new Map();
const context=vm.createContext({document:{getElementById(id){
  if(!elements.has(id)) elements.set(id,{}); return elements.get(id);
},querySelectorAll(){return []}},fetch:()=>new Promise(()=>{}),setInterval(){},localStorage:{}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
for(const memory of [null, undefined, 98304]){
  context.memory=memory;
  vm.runInContext(`render({inventory:{demos:[],gpus:[
    {name:'NVIDIA GB10',memory_mib:memory,nvenc:true}],driver_version:'580.1'},
    process:{state:'stopped'}})`,context);
  const summary=elements.get('summary').innerHTML;
  assert.match(summary,/NVIDIA GB10/);
  assert.match(summary,/NVENC: Yes/);
  assert.match(summary,memory==null ? /Memory unavailable/ : /96.0 GiB/);
  assert.doesNotMatch(summary,/0.0 GiB|NaN|No GPU detected/);
}
""", str(page)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_release_selector_and_recording_download_links(self):
        page = Path(__file__).resolve().parents[2] / "deployments/brev/ui.html"
        result = subprocess.run(["node", "-e", """
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const elements=new Map();
const context=vm.createContext({document:{getElementById(id){
  if(!elements.has(id)) elements.set(id,{value:'',innerHTML:''}); return elements.get(id);
},querySelectorAll(){return []}},fetch:()=>new Promise(()=>{}),setInterval(){},localStorage:{}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],context);
(async()=>{
  vm.runInContext(`api=async(path)=>path==='/api/releases'
    ? {tags:['v0.3.0-beta','v0.2.0-beta.2'],latest:'v0.3.0-beta'}
    : {recordings:[{id:'simple-vlm/123',participant:'<script>',complete:true}]};
    act=(path,body)=>{globalThis.sent={path,body}};`,context);
  await vm.runInContext('loadReleases()',context);
  assert.match(elements.get('version').innerHTML,/Latest release \\(v0.3.0-beta\\)/);
  assert.match(elements.get('version').innerHTML,/v0.2.0-beta.2/);
  for(const version of ['latest','v0.3.0-beta','v0.2.0-beta.2','current']){
    elements.get('version').value=version;
    elements.get('load').onclick();
    assert.equal(context.sent.body,'c='+encodeURIComponent(version));
  }
  elements.get('c').value='a'.repeat(40);
  elements.get('load').onclick();
  assert.equal(context.sent.body,'c='+'a'.repeat(40));
  elements.get('version').value='current';
  elements.get('version').onchange();
  elements.get('load').onclick();
  assert.equal(context.sent.body,'c=current');
  vm.runInContext(`render({inventory:{revision:'v0.3.0-beta',commit:'abc123456789abcdef',
    demos:[],gpus:[]},process:{state:'stopped'}})`,context);
  assert.match(elements.get('summary').innerHTML,/class="value">v0.3.0-beta/);
  assert.match(elements.get('summary').innerHTML,/abc123456789/);
  await vm.runInContext('loadRecordings()',context);
  assert.match(elements.get('recordings').innerHTML,/api\\/recording\\?id=simple-vlm%2F123/);
  assert.match(elements.get('recordings').innerHTML,/&lt;script&gt;/);
  assert.doesNotMatch(elements.get('recordings').innerHTML,/<script>/);
})().catch(error=>{console.error(error);process.exitCode=1});
""", str(page)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recording_selection_redeploy_and_token_restart(self):
        page = Path(__file__).resolve().parents[2] / "deployments/brev/ui.html"
        result = subprocess.run(["node", "-e", """
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const elements = new Map();
const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {value:'', showModal() {}});
      return elements.get(id);
    }, querySelectorAll() {return []},
  }, localStorage: {}, fetch: () => new Promise(() => {}), setInterval() {},
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').split('<script>')[1].split('</script>')[0], context);
vm.runInContext(`let status = {
  inventory:{gpus:[], recommended:'simple-vlm', demos:[
    {id:'simple-vlm', runnable:true, recording_flag:'--session-recording'},
    {id:'legacy', runnable:true, recording_flag:'--capture'},
    {id:'procedure', runnable:true, recording_flag:''}]},
  process:{state:'ready', sample:'simple-vlm', recording:false, anonymous:false}
}; render(status); act=(path,body)=>{globalThis.sent={path,body}};`, context);
assert.equal(elements.get('recording').checked, false);
assert.equal(elements.get('start').disabled, true);
elements.get('recording').checked = true;
elements.get('recording').onchange();
assert.equal(elements.get('start').disabled, false);
elements.get('start').onclick();
assert.equal(context.sent.body, 'sample=simple-vlm&recording=1');
context.document.getElementById('ft').value = 'hf_test';
elements.get('use').onclick();
assert.equal(context.sent.body, 'sample=simple-vlm&recording=1&h=hf_test');
vm.runInContext('status.process.recording=true; render(status)', context);
assert.equal(elements.get('start').disabled, true);
elements.get('recording').checked = false;
elements.get('recording').onchange();
assert.equal(elements.get('start').disabled, false);
elements.get('start').onclick();
assert.equal(context.sent.body, 'sample=simple-vlm');
vm.runInContext("selected='legacy'; recording=true; render(status)", context);
elements.get('start').onclick();
assert.equal(context.sent.body, 'sample=legacy&recording=1');
vm.runInContext("selected='procedure'; render(status)", context);
assert.equal(elements.get('recording').disabled, true);
assert.equal(elements.get('recording').checked, false);
elements.get('start').onclick();
assert.equal(context.sent.body, 'sample=procedure');
""", str(page)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_removed_log_source_falls_back_to_combined(self):
        page = Path(__file__).resolve().parents[2] / "deployments/brev/ui.html"
        result = subprocess.run(["node", "-e", """
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const elements = new Map();
const context = vm.createContext({
  document: {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value: '', innerHTML: ''});
    return elements.get(id);
  }},
  fetch: () => new Promise(() => {}), setInterval() {},
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').split('<script>')[1].split('</script>')[0], context);
(async () => {
  context.calls = [];
  context.sources = [{id:'combined', name:'Combined'}, {id:'old/worker.log', name:'Worker'}];
  vm.runInContext(`api=async (path)=>{
    calls.push(path);
    if(path==='/api/logs') return {sources};
    return {text:'new output', offset:10};
  }; source='old/worker.log'; offset=45; text='previous output';`, context);
  await vm.runInContext('logs()', context);
  assert.equal(context.calls[1], '/api/log?id=old%2Fworker.log&offset=45');
  context.sources = [{id:'combined', name:'Combined'}, {id:'new/hub.log', name:'Hub'}];
  context.calls = [];
  await vm.runInContext('logs()', context);
  assert.equal(context.calls[1], '/api/log?id=combined&offset=0');
  assert.equal(vm.runInContext('source', context), 'combined');
  assert.equal(vm.runInContext('text', context), 'new output');
  assert.match(elements.get('logSource').innerHTML, /value="combined" selected/);
  assert.doesNotMatch(elements.get('logSource').innerHTML, /old\\/worker/);
})().catch(error => {console.error(error); process.exitCode = 1;});
""", str(page)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_token_entry_while_stopped_deploys_selected_sample(self):
        page = Path(__file__).resolve().parents[2] / "deployments/brev/ui.html"
        result = subprocess.run(["node", "-e", """
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const elements = new Map();
const context = vm.createContext({
  document: {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {hidden: false, value: '', showModal() {this.open = true}});
      return elements.get(id);
    },
    querySelectorAll() {return []},
  },
  localStorage: {}, fetch: () => new Promise(() => {}), setInterval() {},
});
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').split('<script>')[1].split('</script>')[0], context);
for (const sample of ['', 'old-sample']) {
  context.previous = sample;
  vm.runInContext(`render({
    inventory: {gpus: [], demos: [{id:'selected-sample', name:'Selected', runnable:true}],
                recommended:'selected-sample'},
    process: {state:'stopped', sample:previous, anonymous:true}
  }); act=(path,body)=>{globalThis.sent={path,body}};`, context);
  assert.equal(elements.get('auth').hidden, false);
  assert.equal(elements.get('auth').textContent, 'Use HF token');
  elements.get('auth').onclick();
  assert.equal(elements.get('tip').open, true);
  context.document.getElementById('ft').value = 'hf_test';
  elements.get('use').onclick();
  assert.equal(context.sent.path, '/api/start');
  assert.equal(context.sent.body, 'sample=selected-sample&h=hf_test');
  assert.equal(elements.get('ft').value, '');
}
""", str(page)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
