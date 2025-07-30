<template>
  <div class="container" style="padding: 20px">
    <h1>🎧 播放端 </h1>

    <a-space style="margin-bottom: 10px">
      <a-button type="primary" @click="loadAudioList">📥 加载音频列表</a-button>
      <a-button @click="startReplay" :disabled="isRunning || audioList.length === 0">▶️ 开始重放</a-button>
      <a-button @click="stopReplay" :disabled="!isRunning">⏹️ 停止重放</a-button>
      <div>{{ status }}</div>
    </a-space>

    <audio ref="audioPlayer" @ended="onAudioEnded" controls style="width: 100%; margin-bottom: 10px;"></audio>

    <a-table :dataSource="audioList" :columns="columns" :pagination="{ pageSize: 50 }" rowKey="id" bordered
      :rowClassName="rowClassName" />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const audioList = ref([])
const currentIndex = ref(0)
const isRunning = ref(false)
const isPaused = ref(false)
const status = ref('等待开始')
const audioPlayer = ref(null)

const serverIP = "192.168.31.98"

let socket = null
let timeout = null
let wakeLock = null

const columns = [
  { title: '序号', dataIndex: 'index', key: 'index', customRender: ({ index }) => index + 1 },
  { title: '文件名', dataIndex: 'name', key: 'name' },
  { title: 'ID', dataIndex: 'id', key: 'id' },
  {
    title: '状态',
    key: 'status',
    customRender: ({ record }) => {
      const idx = audioList.value.findIndex(a => a.id === record.id)
      if (idx < currentIndex.value) return '✅ 已播放'
      if (idx === currentIndex.value && isRunning.value) return '▶️ 正在播放'
      return '⏸ 未播放'
    }
  }
]

function rowClassName(record) {
  const idx = audioList.value.findIndex(a => a.id === record.id)
  return idx === currentIndex.value && isRunning.value ? 'current-row' : ''
}

// init socket on mounted
onMounted(() => {
  initSocket()
  requestWakeLock()
  loadAudioList()
})

async function requestWakeLock() {
  if ('wakeLock' in navigator) {
    this.wakeLock = await navigator.wakeLock.request('screen');
  } else {
    console.warn('当前浏览器不支持屏幕唤醒功能')
  }
}

function initSocket() {
  socket = new WebSocket(`wss://${serverIP}:8443`)
  socket.onopen = () => {
    socket.send(JSON.stringify({ role: 'A' }));
    console.log("WebSocket connected")
  }
  socket.onerror = (error) => {
    console.error("WebSocket error:", error)
    status.value = "WebSocket 连接失败"
  }
  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    if (msg.type === "ready" && msg.id === currentAudio().id) {
      clearTimeout(timeout) // 清除超时计时器
      playCurrentAudio()
    }
    if (msg.type === "record_done" && msg.id === currentAudio()?.id && isRunning.value) {
      status.value = `录音保存成功，准备下一个`
      currentIndex.value++
      // replayNext()
      isRunning.value = false
      setTimeout(() => {
        isRunning.value = true
        replayNext()
      }, 1500)
    }
    if (msg.type === "error") {
      status.value = `错误：${msg.message}`
      isRunning.value = false
    }
  }
}

function loadAudioList() {
  axios.get(`https://${serverIP}:3443/audio-list`)
    .then(res => {
      audioList.value = res.data
      status.value = `已加载 ${audioList.value.length} 条音频`
    })
    .catch(err => {
      status.value = '加载音频列表失败'
      console.error(err)
    })
}

function currentAudio() {
  return audioList.value[currentIndex.value]
}

function startReplay() {
  if (!socket) initSocket()
  isRunning.value = true
  currentIndex.value = 0
  replayNext()
}

function replayNext() {
  if (currentIndex.value >= audioList.value.length) {
    status.value = '重放完成'
    isRunning.value = false
    this.wakeLock?.release()

    return
  }
  const audio = currentAudio()
  status.value = `准备第 ${currentIndex.value + 1} 条：${audio.name}`

  // 发送开始录制的消息
  socket.send(JSON.stringify({
    type: "start_record",
    id: audio.id,
    filename: audio.name,
  }))

  // playCurrentAudio()
  // 设置超时机制
  timeout = setTimeout(() => {
    status.value = `播放失败：未收到准备成功消息`
    isRunning.value = false

  }, 3000) // 3秒超时
}

function playCurrentAudio() {
  const audio = currentAudio()
  audioPlayer.value.src = `https://${serverIP}:3443/audio/${audio.name}`
  audioPlayer.value.play()
  status.value = `正在播放：${audio.name}`
}


function stopReplay() {
  status.value = '已停止重放'
  isRunning.value = false
  isPaused.value = false
  if (audioPlayer.value) {
    audioPlayer.value.pause()
  }
  const audio = currentAudio()
  if (socket && audio) {
    socket.send(JSON.stringify({
      type: "end",
      id: audio.id
    }))
  }
}

function onAudioEnded() {
  const audio = currentAudio()
  status.value = `播放完成，等待接收端上传录音`
  socket.send(JSON.stringify({
    type: 'end',
    id: audio.id
  }))
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

.controls {
  margin-top: 10px;
  margin-bottom: 10px;
  display: flex;
  gap: 10px;
}

button {
  padding: 6px 12px;
}
</style>