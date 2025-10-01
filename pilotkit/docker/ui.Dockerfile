FROM node:18-alpine

WORKDIR /app
COPY ui/package.json ui/package-lock.json* ./
RUN npm install
COPY ui /app
RUN npm run build
CMD ["npm", "run", "start"]
