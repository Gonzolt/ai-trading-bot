FROM node:20-alpine AS build
WORKDIR /app
COPY services/dashboard/ui/package*.json ./
RUN npm install
COPY services/dashboard/ui/ ./
ARG VITE_API_BASE_URL=/api
ARG VITE_WS_BASE_URL=/ws/live
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL VITE_WS_BASE_URL=$VITE_WS_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY docker/nginx/ui.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
