// websocket_server.js

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const https = require('https');

const PORT = 8443;
const SAVE_DIR = path.join(__dirname, 'recordings');

if (!fs.existsSync(SAVE_DIR)) fs.mkdirSync(SAVE_DIR, { recursive: true });

const options = {
  key: fs.readFileSync('./localhost-key.pem'),
  cert: fs.readFileSync('./localhost.pem'),
};

const server = https.createServer({
  key: fs.readFileSync('localhost-key.pem'),
  cert: fs.readFileSync('localhost.pem')
});
const wss = new WebSocket.Server({ server });

let clientA = null;
let clientB = null;

console.log("Waiting for clients to connect...");

wss.on('connection', (ws) => {
//   console.log("New connection");

  ws.on('message', (data) => {
    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch (e) {
      console.error("Invalid JSON message", e);
      return;
    }
    

    if (msg['role'] === 'A' && !clientA) {
      clientA = ws;
      ws.role = 'A';
      console.log("Client A connected");
    } else if ((!clientB && ws !== clientA) || msg['role'] === 'B') {
      clientB = ws;
      ws.role = 'B';
      console.log("Client B connected");
    }

    // 控制信息中转
    if (ws.role === 'A' && clientB) {
      // A 发送的控制信号转发给 B
      clientB.send(JSON.stringify(msg));
    } else if (ws.role === 'B' && clientA) {

      // B 发送的信息转发给 A
      console.log("sending to A:", msg);
      clientA.send(JSON.stringify(msg));
    }
  });

  ws.on('close', () => {
    console.log(`${ws.role || "Unknown client"} disconnected`);
    if (ws === clientA) clientA = null;
    if (ws === clientB) clientB = null;
  });
});

server.listen(PORT, () => {
  console.log(`HTTPS + WebSocket Server 已启动： https://localhost:${PORT}`);
});

const express = require('express');
const cors = require('cors');
const app = express();
const multer = require('multer')
const upload = multer({ dest: 'recordings/' })

app.use(cors());

app.get('/audio-list', (req, res) => {
  const audioDir = path.join(__dirname, 'audio'); // 放音频的目录
  console.log("Audio directory:", audioDir);
  fs.readdir(audioDir, (err, files) => {
    if (err) return res.status(500).json({ error: err.message });
    const audioFiles = files.filter(f => f.endsWith('.wav') || f.endsWith('.mp3')).map((f, i) => ({
      id: String(i).padStart(3, '0'),
      name: f,
      url: `/audio/${f}`
    }));
    res.json(audioFiles);
    // console.log(audioFiles);
  });
});

app.use('/audio', express.static(path.join(__dirname, 'audio'))); // 静态服务

app.post('/upload', upload.single('audio'), (req, res) => {
  console.log("Received file upload request");
  let { id, filename } = req.body;
  if (!req.file) {
    return res.status(400).send('未收到文件');
  }
  if (!id || !filename) {
    return res.status(400).send('缺少 id 或 filename');
  }
  const tempPath = req.file.path;
  filename = filename.replace(".mp3", ".wav")
  const targetPath = path.join(__dirname, 'recordings', `${filename}`);

  fs.rename(tempPath, targetPath, err => {
    if (err) return res.status(500).send('保存失败');
    res.status(200).send('保存成功');
  });
});

// app.listen(3000, '0.0.0.0', () => {
//   console.log('HTTP server on http://localhost:3000');
// });
const httpsServer = https.createServer(options, app);
httpsServer.listen(3443, '0.0.0.0', () => {
  console.log('HTTPS server on https://localhost:3443');
});