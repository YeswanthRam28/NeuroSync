import streamlit as st
import streamlit.components.v1 as components

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>NeuroSync Gesture Racing Demo</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.6.0/p5.min.js"></script>

  <style>
    body {
      margin: 0;
      padding: 0;
      background: #111;
      overflow: hidden;
      font-family: Arial, sans-serif;
      color: white;
    }
    #status {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.5);
      padding: 10px 15px;
      border-radius: 10px;
      font-size: 14px;
    }
  </style>
</head>
<body>

<div id="status">Connecting to gesture server...</div>

<script>
let car;
let accelerate = 0;
let steer = 0;
let brake = 0;
let boost = 0;

// ----------------------
// WEBSOCKET CONNECTION
// ----------------------
let ws = new WebSocket("ws://127.0.0.1:8765/ws");

ws.onopen = () => {
  document.getElementById("status").textContent = "Connected! Move your hand 🙌";
  setInterval(() => ws.send("ping"), 1000);
};

ws.onerror = () => {
  document.getElementById("status").textContent = "WS error (is server running?)";
};

ws.onclose = () => {
  document.getElementById("status").textContent = "WS closed";
};

ws.onmessage = (event) => {
  try {
    const data = JSON.parse(event.data);

    if (data.accelerate !== undefined) {
      accelerate = data.accelerate;
      brake = data.brake;
      steer = data.steer;
      boost = data.boost;
    }

  } catch (e) {
    console.error("WS message error:", e);
  }
};

// ----------------------
// CAR CLASS
// ----------------------
class Car {
  constructor() {
    this.x = window.innerWidth / 2;
    this.y = window.innerHeight - 120;

    this.angle = 0;
    this.speed = 0;
    this.maxSpeed = 10;
    this.accelRate = 0.2;
    this.brakeRate = 0.4;
  }

  update() {
    // apply gesture acceleration
    if (accelerate > 0.1) {
      this.speed += this.accelRate * accelerate;
    }

    // brake
    if (brake === 1) {
      this.speed -= this.brakeRate;
    }

    // boost
    if (boost === 1) {
      this.speed += 0.6;
    }

    // apply friction
    this.speed *= 0.98;

    // limit speed
    this.speed = constrain(this.speed, 0, this.maxSpeed);

    // steering from gesture
    this.angle += steer * 2;

    // update position
    this.x += Math.sin(radians(this.angle)) * this.speed;
    this.y -= Math.cos(radians(this.angle)) * this.speed;

    // wrap edges
    if (this.x < 0) this.x = width;
    if (this.x > width) this.x = 0;
    if (this.y < 0) this.y = height;
    if (this.y > height) this.y = 0;
  }

  draw() {
    push();
    translate(this.x, this.y);
    rotate(radians(this.angle));
    rectMode(CENTER);

    // glowing car body
    fill(255, 80, 80);
    rect(0, 0, 40, 70, 10);

    // headlights
    fill(255, 255, 180);
    rect(0, -40, 25, 10, 4);

    pop();
  }
}

// ----------------------
// P5 SETUP + DRAW LOOP
// ----------------------
function setup() {
  createCanvas(window.innerWidth, window.innerHeight);
  car = new Car();
}

function draw() {
  background(20);

  // road effect
  stroke(60);
  for (let i = 0; i < height; i += 40) {
    line(width / 2, i, width / 2, i + 20);
  }

  car.update();
  car.draw();

  // UI overlay
  fill(255);
  textSize(15);
  text(`Accel: ${accelerate.toFixed(2)}`, 20, 40);
  text(`Steer: ${steer.toFixed(2)}`, 20, 60);
  text(`Brake: ${brake}`, 20, 80);
  text(`Boost: ${boost}`, 20, 100);
}
</script>

</body>
</html>

"""

components.html(html_code, height=600)
