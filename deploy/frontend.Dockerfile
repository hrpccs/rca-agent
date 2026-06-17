# Frontend image: build the Vite app, serve static assets via nginx with an API
# proxy to the backend. Build context = repo root (so we can reach deploy/ and
# frontend/); the worker-owned frontend/ dir is the app source.
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# nginx-unprivileged runs as non-root (UID 101) and listens on 8080 by default,
# matching `listen 8080` in frontend.nginx.conf.
FROM nginxinc/nginx-unprivileged:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/frontend.nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
