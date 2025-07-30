# Relay Data Construction Tools

Change `serverIP` to your server ip in `client/player/src/App.vue` and `client/recorder/src/App.vue` before run

```bash
# Step 1: Configure local https
mkcert -install
mkcert localhost
```
Then copy the generated `localhost.pem` and `localhost-key.pem` to `server/`, `client/player` and `client/recorder`

```bash
# Step 2: Run player
cd client/player
npm install
npm run dev
```

```bash
# Step 3: Run recorder
cd client/recorder
npm install
npm run dev
```

Put your audio files that need to be replayed in `server/audio` directory. Replayed audio files will be stored in `server/recordings` directory.

```bash
# Step 4: Run server
cd server
node websocket_server.js
```