FROM node:20-alpine

WORKDIR /app
COPY apps/web/package*.json ./
RUN npm install

COPY apps/web/ .
RUN npm run build

CMD ["npm", "start"]
