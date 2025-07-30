<template>
  <div class="container" style="padding: 20px">
    <h1>🎙 接收端</h1>
    <a-button type="primary" @click="getPermission" style="margin-bottom: 16px">获取麦克风权限</a-button>
    <a-alert :message="status" type="info" show-icon style="margin-bottom: 16px" />
    <a-select v-model="selectedDeviceId" placeholder="选择录音设备" style="width: 300px; margin-bottom: 16px"
      :options="audioInputOptions" />

  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import axios from 'axios'

const status = ref('等待控制信号')

let socket = null

let currentId = ''
let currentFilename = ''

let stopCurrentRecording = null

const serverIP = "192.168.31.98"

let recorder = null
let audioContext = null
let gumStream = null
let inputNode = null

const selectedDeviceId = ref(null)
const audioInputOptions = ref([])

onMounted(() => {

  connect()
  requestWakeLock()
  listAudioDevices()
  initRecorderOnce()
  stopCurrentRecording = () => {
    recorder.stop()
    recorder.exportWAV(blob => {
      uploadRecording(blob, currentId, currentFilename)
      recorder.clear()
    })
  }

})

function connect() {
  socket = new WebSocket(`wss://${serverIP}:8443`)
  socket.onopen = () => {
    socket.send(JSON.stringify({ role: 'B' }));
    status.value = "已连接到服务器"
  }
  socket.onerror = (error) => {
    console.error("WebSocket error:", error)
    status.value = "WebSocket 连接失败"
    return
  }

  socket.onmessage = handleSocketMessage
}

function getPermission() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      status.value = "麦克风权限已获取"
      message.success('麦克风权限已获取')
      stream.getTracks().forEach(track => track.stop()) // 停止流
    })
    .catch(err => {
      status.value = "麦克风权限获取失败"
      message.error('麦克风权限获取失败：' + err.message)
    })
}

async function listAudioDevices() {
  try {
    // 先请求权限，才能获取 device label
    await navigator.mediaDevices.getUserMedia({ audio: true })
    const devices = await navigator.mediaDevices.enumerateDevices()
    audioInputOptions.value = devices
      .filter(d => d.kind === 'audioinput')
      .map(d => ({ label: d.label || '未知设备', value: d.deviceId }))
  } catch (err) {
    message.error('获取音频设备失败：' + err.message)
  }
}


function requestWakeLock() {
  if ('wakeLock' in navigator) {
    navigator.wakeLock.request('screen').then(() => {
      console.log('屏幕保持唤醒状态')
    }).catch(err => {
      console.error('请求屏幕唤醒失败:', err)
    })
  } else {
    console.warn('当前浏览器不支持屏幕唤醒功能')
  }
}

async function initRecorderOnce() {
  if (recorder) return
  let constraints = { audio: true }
  if (selectedDeviceId.value) {
    constraints = {
      audio: {
        deviceId: { exact: selectedDeviceId.value }
      }
    }
  }
  gumStream = await navigator.mediaDevices.getUserMedia(constraints)
  audioContext = new (window.AudioContext || window.webkitAudioContext)()
  inputNode = audioContext.createMediaStreamSource(gumStream)
  recorder = new Recorder(inputNode, { numChannels: 1 })
}

function handleSocketMessage(event) {
  const msg = JSON.parse(event.data)
  // console.log(msg)

  if (msg.type === 'start_record') {
    currentId = msg.id
    currentFilename = msg.filename
    startRecording()
  }

  if (msg.type === 'end') {
    stopCurrentRecording()
  }
}
function startRecording() {
  console.log("starting")
  recorder.record()
  status.value = `正在录音：${currentFilename}`

  // 回复 ready
  socket.send(JSON.stringify({
    type: 'ready',
    id: currentId
  }))
}
// function startRecording() {
//   navigator.mediaDevices.getUserMedia({
//     audio: true
//     // {
//     //   echoCancellation: false,
//     //   noiseSuppression: false,
//     //   autoGainControl: false
//     // }
//   }).then(stream => {

//     const audioContext = new (window.AudioContext || window.webkitAudioContext)()
//     const input = audioContext.createMediaStreamSource(stream)
//     const recorder = new Recorder(input, { numChannels: 1 }) // mono

//     recorder.record()

//     status.value = `正在录音：${currentFilename}`

//     stopCurrentRecording = () => {
//       recorder.stop()
//       recorder.exportWAV(blob => {
//         uploadRecording(blob, currentId, currentFilename)

//         stream.getTracks().forEach(track => track.stop())
//         audioContext.close()
//         recorder.clear()
//         audioContext = null
//         input = null
//         recorder = null

//         stopCurrentRecording = null // 清理引用

//       })
//     }

//     // 回复 ready
//     socket.send(JSON.stringify({
//       type: 'ready',
//       id: currentId
//     }))
//   }).catch(err => {
//     message.error('录音失败：' + err.message)
//   })
// }


function uploadRecording(blob, id, filename) {
  let formData = new FormData()
  formData.append('id', id)
  formData.append('filename', filename)
  formData.append('audio', blob)

  axios.post(`https://${serverIP}:3443/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }).then(res => {
    if (res.status === 200) {
      // message.success('录音上传成功')
      status.value = `录音上传成功：${filename}`
      socket.send(JSON.stringify({
        type: 'record_done',
        id: id
      }))
    } else {
      message.error('录音上传失败')
      socket.send(JSON.stringify({
        type: 'error',
        id: id
      }))
      // stopRecording() // 停止录音
    }
  }).catch(err => {
    message.error('录音上传失败：' + err.message)
  })
}

</script>

<style scoped>
.container {
  padding: 20px;
  font-family: sans-serif;
  font-size: 18px;
  width: 90vw;
  height: 90vh;
}

#waveform {
  background-color: #f9f9f9;
}
</style>