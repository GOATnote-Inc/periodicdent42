FROM node:20-alpine
WORKDIR /app
COPY ui/package.json package.json
RUN npm install
COPY ui /app
ENV NEXT_PUBLIC_API_BASE=http://orchestrator:8080
CMD ["npm", "run", "dev", "--", "-p", "3000", "-H", "0.0.0.0"]
