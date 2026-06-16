# Frontend image: build the Vite app, serve static assets via nginx with an API
# proxy to the backend. Build context = repo root (so we can reach deploy/ and
# frontend/); the worker-owned frontend/ dir is the app source.
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/frontend.nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
