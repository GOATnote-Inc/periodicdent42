FROM node:20-alpine
WORKDIR /app
COPY ui/package.json ui/package-lock.json* ./
RUN npm install
COPY ui ./
RUN npm run build
CMD ["npm", "run", "start"]
