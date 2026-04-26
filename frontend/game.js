/**
 * game.js
 * Solar EV Sensor Window & Tactical Overview
 */

// Detect Backend URL: If hosted on Vercel/External, point to the Hugging Face Space.
const PROD_BACKEND = "https://debug180906-solar-ev-env.hf.space";
const BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.origin === PROD_BACKEND) 
    ? window.location.origin 
    : PROD_BACKEND;

// State
let tasks = {};
let currentTask = null;
let state = null;
let isAutoDriving = false;
let autoDriveTimer = null;

// Track Geometry
let trackPoints = []; // [{x, y, km, incline, solar}]
let totalTrackDist = 0;

// Visual interpolation state
let renderState = {
    x: 0,
    speed: 0,
    lastTime: performance.now(),
};

// DOM
const sensorCanvas = document.getElementById('sensorCanvas');
const circuitCanvas = document.getElementById('circuitCanvas');
const ctxS = sensorCanvas.getContext('2d');
const ctxC = circuitCanvas.getContext('2d');
const taskSelect = document.getElementById('task-select');
const btnReset = document.getElementById('btn-reset');
const btnStep = document.getElementById('btn-step');
const btnAuto = document.getElementById('btn-auto');

function resize() {
    sensorCanvas.width = sensorCanvas.parentElement.clientWidth;
    sensorCanvas.height = sensorCanvas.parentElement.clientHeight;
    circuitCanvas.width = circuitCanvas.parentElement.clientWidth;
    circuitCanvas.height = circuitCanvas.parentElement.clientHeight;
}
window.addEventListener('resize', resize);
resize();

// ----------------------------------------------------
// 1. Math & Spline Generation
// ----------------------------------------------------
const SHAPES = {
    'flat_track_easy': [{x:0,y:0}, {x:10,y:0}, {x:10,y:5}, {x:0,y:5}],
    'dynamic_routing_medium': [{x:0,y:0}, {x:10,y:5}, {x:5,y:10}, {x:-5,y:5}],
    'night_run_no_solar': [{x:0,y:0}, {x:5,y:-5}, {x:10,y:0}, {x:5,y:10}, {x:0,y:5}, {x:-5,y:10}],
    'thermal_race_hard': [{x:0,y:0}, {x:10,y:0}, {x:12,y:5}, {x:8,y:8}, {x:15,y:12}, {x:0,y:10}, {x:-5,y:5}],
    'ultra_endurance_expert': [{x:0,y:0}, {x:10,y:-5}, {x:20,y:0}, {x:25,y:10}, {x:15,y:20}, {x:5,y:15}, {x:-10,y:20}, {x:-15,y:10}]
};

function getCatmullRom(t, points) {
    const p0 = points[(Math.floor(t) - 1 + points.length) % points.length];
    const p1 = points[Math.floor(t) % points.length];
    const p2 = points[(Math.floor(t) + 1) % points.length];
    const p3 = points[(Math.floor(t) + 2) % points.length];
    const f = t - Math.floor(t);
    const f2 = f * f; const f3 = f2 * f;
    const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * f + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * f2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * f3);
    const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * f + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * f2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * f3);
    return { x, y };
}

function buildTrack(task) {
    const shape = SHAPES[task.task_id] || SHAPES['flat_track_easy'];
    const numSamples = 1000;
    let rawPoints = [];
    let cumulativeDist = 0;
    
    // Generate raw spline points
    for (let i = 0; i < numSamples; i++) {
        let t = (i / numSamples) * shape.length;
        let pt = getCatmullRom(t, shape);
        if (rawPoints.length > 0) {
            let last = rawPoints[rawPoints.length-1];
            cumulativeDist += Math.hypot(pt.x - last.x, pt.y - last.y);
        }
        pt.rawDist = cumulativeDist;
        rawPoints.push(pt);
    }
    
    // Scale to task km
    const targetKm = task.total_distance_km;
    const scale = targetKm / cumulativeDist;
    trackPoints = rawPoints.map(p => ({
        x: p.x * scale * 100, // 100px per km
        y: p.y * scale * 100,
        km: p.rawDist * scale
    }));
    totalTrackDist = targetKm;
    
    // Attach segment data and calculate elevation
    let currentSegIdx = 0;
    let segStart = 0;
    let currentElev = 0;
    
    trackPoints.forEach((p, i) => {
        let seg = task.segments[currentSegIdx];
        if (seg && p.km > segStart + seg.distance_km) {
            segStart += seg.distance_km;
            currentSegIdx++;
            seg = task.segments[currentSegIdx] || seg;
        }
        p.incline = seg ? seg.incline_pct : 0;
        p.solar = seg ? seg.solar_irradiance_wm2 : 0;
        
        if (i > 0) {
            let dKm = p.km - trackPoints[i-1].km;
            // incline is %, so incline/100 * dKm gives dElev in km. * 1000 = meters
            currentElev += (trackPoints[i-1].incline / 100) * (dKm * 1000);
        }
        p.elevation = currentElev;
    });
}

function getPosAtKm(km) {
    km = Math.max(0, Math.min(km, totalTrackDist)); 
    for (let i = 0; i < trackPoints.length - 1; i++) {
        if (km >= trackPoints[i].km && km <= trackPoints[i+1].km) {
            let range = trackPoints[i+1].km - trackPoints[i].km;
            let t = range === 0 ? 0 : (km - trackPoints[i].km) / range;
            return {
                x: trackPoints[i].x + t * (trackPoints[i+1].x - trackPoints[i].x),
                y: trackPoints[i].y + t * (trackPoints[i+1].y - trackPoints[i].y),
                elevation: trackPoints[i].elevation + t * (trackPoints[i+1].elevation - trackPoints[i].elevation),
                incline: trackPoints[i].incline
            };
        }
    }
    return trackPoints[0];
}

function getAngleAtKm(km) {
    let p1 = getPosAtKm(km);
    let p2 = getPosAtKm(km + 0.1);
    return Math.atan2(p2.y - p1.y, p2.x - p1.x);
}

// ----------------------------------------------------
// 2. API & Logic
// ----------------------------------------------------
async function apiGet(path) {
    const r = await fetch(`${BASE_URL}${path}`);
    return r.json();
}
async function apiPost(path, body = null) {
    const opts = { method: 'POST' };
    if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
    }
    const r = await fetch(`${BASE_URL}${path}`, opts);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}

function setStatus(msg) {
    document.getElementById('app-status').innerText = msg;
}

async function init() {
    setStatus("Fetching task definitions...");
    const data = await apiGet('/tasks');
    data.tasks.forEach(t => {
        tasks[t.task_id] = t;
        const opt = document.createElement('option');
        opt.value = t.task_id;
        opt.innerText = `${t.name} (${t.difficulty})`;
        taskSelect.appendChild(opt);
    });
    taskSelect.options[0].remove();
    await resetEpisode();
    requestAnimationFrame(gameLoop);
}

async function resetEpisode() {
    stopAutoDrive();
    document.getElementById('banner').classList.add('hidden');
    
    const taskId = taskSelect.value;
    currentTask = tasks[taskId];
    buildTrack(currentTask);
    
    const isRnd = document.getElementById('check-randomize').checked;
    const seed = document.getElementById('input-seed').value;
    
    let url = `/api/reset?task_id=${taskId}&randomize=${isRnd}`;
    if (seed) url += `&seed=${parseInt(seed)}`;
    
    setStatus(`Initializing ${taskId}...`);
    try {
        state = { observation: await apiPost(url) };
        renderState.x = 0;
        renderState.speed = 0;
        updateHUD();
        fetchAdvisor();
        setStatus("SYSTEM READY");
        document.getElementById('mission-log').innerText = "Episode started.";
    } catch(e) {
        setStatus(`ERROR: ${e.message}`);
    }
}

async function takeStep(action) {
    if (state?.reward?.is_done) return;
    setStatus("Executing step...");
    try {
        const result = await apiPost('/api/step', action);
        state = result;
        updateHUD();
        fetchAdvisor();
        
        if (state.reward.is_done) {
            showBanner(state.reward.is_success, state.reward.score);
            stopAutoDrive();
        }
        setStatus("SYSTEM READY");
    } catch(e) {
        setStatus(`ERROR: ${e.message}`);
        stopAutoDrive();
    }
}

async function fetchAdvisor() {
    try {
        const forecast = await apiPost('/advisor');
        document.getElementById('advisor-forecast').innerText = forecast.reasoning;
        
        const cs = document.getElementById('chip-solar');
        cs.innerText = `ENERGY: ${forecast.energy_risk.toUpperCase()}`;
        cs.className = `risk-chip ${forecast.energy_risk === 'critical' ? 'high' : (forecast.energy_risk === 'high' ? 'med' : 'low')}`;
        
        const ct = document.getElementById('chip-temp');
        ct.innerText = `THERM: ${forecast.thermal_risk.toUpperCase()}`;
        ct.className = `risk-chip ${forecast.thermal_risk === 'critical' ? 'high' : (forecast.thermal_risk === 'high' ? 'med' : 'low')}`;
    } catch(e) {
        document.getElementById('advisor-forecast').innerText = "Link failed.";
    }
}

// ----------------------------------------------------
// 3. Task-Aware Strategy (Autopilot)
// ----------------------------------------------------
function getTaskAwareAction() {
    if (!state) return null;
    const v = state.observation.vehicle;
    const s = state.observation.segment_ahead;
    const taskId = currentTask.task_id;
    
    let speed = 60.0;
    let cooling = 1;
    let routing = "direct_to_motor";
    let thought = "CRUISING NOMINALLY.";

    // Task-specific overrides
    if (taskId === 'night_run_no_solar') {
        speed = 40.0; // Conserve
        thought = "NIGHT OP: SPEED CAPPED. CONSERVING ENERGY.";
    } else if (taskId === 'thermal_race_hard') {
        // Pre-cool MUCH earlier - at 32°C not 38°C
        if (v.battery_temp_c > 32.0) cooling = 2;
        // Emergency slowdown at 40°C
        if (v.battery_temp_c > 40.0) { cooling = 2; speed = Math.min(speed, 35.0); }
        // CRITICAL emergency at 48°C
        if (v.battery_temp_c > 48.0) { cooling = 2; speed = 25.0; }
        // ABSOLUTE emergency at 54°C (4 degrees before failure)
        if (v.battery_temp_c > 54.0) { cooling = 2; speed = 15.0; }
        thought = "THERMAL RACE: AGGRESSIVE COOLING ACTIVE. SPEED LIMITED FOR SURVIVAL.";
    } else if (taskId === 'ultra_endurance_expert') {
        if (v.battery_temp_c > 35.0) cooling = 2;
        if (v.battery_temp_c > 42.0) { cooling = 2; speed = Math.min(speed, 40.0); }
        if (v.battery_temp_c > 50.0) { cooling = 2; speed = 30.0; }
        if (v.battery_temp_c > 55.0) { cooling = 2; speed = 20.0; }
        thought = "ENDURANCE: SURVIVAL MODE. PRESERVING BATTERY AT ALL COSTS.";
    } else if (taskId === 'dynamic_routing_medium') {
        if (v.battery_temp_c > 38.0) cooling = 2;
        if (v.battery_temp_c > 45.0) { cooling = 2; speed = Math.min(speed, 40.0); }
        if (v.battery_temp_c > 52.0) { cooling = 2; speed = 30.0; }
    }
    
    // Immediate tactical overrides
    if (s.average_incline_pct > 3.0) {
        // Uphill = massive heat generation
        cooling = Math.max(cooling, 2);  // Force max cooling
        speed = Math.min(speed, 40.0);    // Cap speed at 40 kph on uphills
        thought += " UPHILL DETECTED: MAX COOLING + SPEED LIMIT.";
    } else if (s.average_incline_pct < -2.0) {
        speed = Math.max(speed, 70.0);
        thought += " DECLINE DETECTED: ENGAGING REGEN BRAKING.";
    }
    
    // Safety Fallbacks matching eval_pipeline.py
    if (v.battery_temp_c > 42.0) cooling = 2;
    if (v.battery_temp_c > 48.0) { speed = Math.min(speed, 40.0); thought = "HIGH TEMP. SPEED REDUCED."; }
    if (v.battery_temp_c > 52.0) { cooling = 2; speed = Math.min(speed, 25.0); thought = "CRITICAL THERMAL! MAX COOLING."; }
    
    // Temperature Trend Detection
    if (!window.tempHistory) window.tempHistory = [];
    window.tempHistory.push(v.battery_temp_c);
    if (window.tempHistory.length > 3) window.tempHistory.shift();

    if (window.tempHistory.length === 3) {
        let tempRise = window.tempHistory[2] - window.tempHistory[0];
        if (tempRise > 2.0) {  // Rising more than 2°C in 2 steps
            cooling = 2;
            speed = Math.min(speed, 35.0);
            thought += " RAPID HEATING: PREVENTIVE SLOWDOWN.";
        }
    }

    if (v.battery_soc_pct < 40.0) speed = Math.min(speed, 45.0);
    if (v.battery_soc_pct < 30.0) { speed = Math.min(speed, 30.0); cooling = Math.min(cooling, 1); }
    if (v.battery_soc_pct < 25.0) {
        speed = Math.min(speed, 20.0);
        cooling = 0;
        thought = "CRITICAL ENERGY! EMERGENCY CONSERVATION.";
    }
    
    document.getElementById('agent-thought').innerText = thought;
    return { target_cruise_speed_kph: speed, cooling_system_level: cooling, solar_routing_mode: routing };
}

function startAutoDrive() {
    if (state?.reward?.is_done) return;
    isAutoDriving = true;
    btnAuto.classList.add('active');
    btnAuto.innerText = "AUTOPILOT ENGAGED";
    
    autoDriveTimer = setInterval(() => {
        if (!state || state.reward?.is_done) return;
        if (Math.abs(renderState.x - state.observation.vehicle.distance_covered_km) < 0.5) {
            takeStep(getTaskAwareAction());
        }
    }, 1200);
}

function stopAutoDrive() {
    isAutoDriving = false;
    btnAuto.classList.remove('active');
    btnAuto.innerText = "ENGAGE AUTOPILOT";
    clearInterval(autoDriveTimer);
    document.getElementById('agent-thought').innerText = "SYSTEM IDLE";
}
btnAuto.addEventListener('click', () => isAutoDriving ? stopAutoDrive() : startAutoDrive());

// ----------------------------------------------------
// 4. UI Updates
// ----------------------------------------------------
function updateHUD() {
    if (!state) return;
    const v = state.observation.vehicle;
    
    document.getElementById('hud-task').innerText = currentTask.name.toUpperCase();
    document.getElementById('hud-dist').innerText = `${v.distance_covered_km.toFixed(1)} / ${currentTask.total_distance_km.toFixed(1)}km`;
    
    // Static values update on step, gameLoop handles the real-time interpolation
    // We keep these here as fallbacks or for non-interpolated fields
    
    // Lookahead
    const upcoming = state.observation.upcoming_segments || [];
    document.getElementById('lookahead-content').innerHTML = upcoming.map(s => `
        <div class="seg-item">
            <span>+${s.distance_to_next_waypoint_km}km</span>
            <span class="seg-inc">${s.average_incline_pct>0?'+':''}${s.average_incline_pct.toFixed(1)}% INC</span>
            <span>${Math.round(s.solar_irradiance_wm2)}W</span>
        </div>
    `).join('');
}

function showBanner(success, score) {
    const b = document.getElementById('banner');
    b.className = ``;
    if(!success) b.classList.add('fail');
    document.getElementById('banner-title').innerText = success ? 'MISSION COMPLETE' : 'SYSTEM FAILURE';
    document.getElementById('banner-sub').innerText = state.reward.feedback;
    document.getElementById('banner-score').innerText = `EVAL SCORE: ${score.toFixed(4)}`;
    
    document.getElementById('mission-log').innerHTML = `
        <strong>FINAL SCORE: ${score.toFixed(4)}</strong><br>
        Status: ${success ? 'SUCCESS' : 'FAILED'}<br><br>
        ${state.reward.feedback}
    `;
}

// ----------------------------------------------------
// 5. Canvas Rendering (Side-Scrolling + Top-Down Circuit)
// ----------------------------------------------------
function drawTrackCircuit(ctx, offset = null) {
    ctx.beginPath();
    ctx.lineWidth = offset ? 40 : 6;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    for (let i = 0; i < trackPoints.length - 1; i++) {
        const p1 = trackPoints[i];
        const p2 = trackPoints[i+1];
        
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        
        let color = '#00d9ff';
        if (currentTask.task_id === 'night_run_no_solar') color = '#6366f1';
        if (currentTask.task_id === 'thermal_race_hard') color = '#ef4444';
        if (p1.incline > 3) color = '#f59e0b';
        if (p1.incline < -2) color = '#10b981';
        
        ctx.strokeStyle = color;
        ctx.stroke();
    }
}

function drawGrid(ctx, w, h) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < w; x += 50) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
    for (let y = 0; y < h; y += 50) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
    ctx.stroke();
}

// Particle system for dirt
let particles = [];
function updateAndDrawParticles(ctx, speed) {
    if (speed > 10 && Math.random() < 0.4) {
        particles.push({
            x: -30 + Math.random()*10, 
            y: 5 + Math.random()*5,
            vx: -Math.random() * 3 - (speed/15), 
            vy: -Math.random() * 2,
            life: 1.0
        });
    }
    for (let i = particles.length - 1; i >= 0; i--) {
        let p = particles[i];
        p.x += p.vx; p.y += p.vy;
        p.vy += 0.15; // gravity
        p.life -= 0.04;
        if (p.life <= 0) {
            particles.splice(i, 1);
            continue;
        }
        ctx.fillStyle = `rgba(100, 80, 60, ${p.life})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2 + Math.random()*2, 0, Math.PI*2);
        ctx.fill();
    }
}

function gameLoop(now) {
    const dt = (now - renderState.lastTime) / 1000;
    renderState.lastTime = now;
    
    if (state && currentTask && trackPoints.length > 0) {
        // Interpolate forward
        const targetDist = state.observation.vehicle.distance_covered_km;
        if (renderState.x < targetDist) {
            // Get actual speed from state
            let actualSpeed = state.observation.vehicle.current_speed_kph || 55.0;
            
            // Convert kph to km/s for visual interpolation
            let kmPerSecond = actualSpeed / 3600.0;
            
            // SUPER ARCADE SPEED: Higher factor + snappy catch-up
            let accelerationFactor = 6.0;  
            let targetVisualSpeed = kmPerSecond * accelerationFactor;
            
            // Gradually accelerate
            renderState.speed = renderState.speed || 0;
            renderState.speed += (targetVisualSpeed - renderState.speed) * 0.3;
            
            // Distance-based catch-up: if more than 50m behind, speed up even more
            let distToTarget = targetDist - renderState.x;
            let moveStep = renderState.speed * dt;
            if (distToTarget > 0.05) moveStep = Math.max(moveStep, distToTarget * 2.0 * dt);
            
            renderState.x += moveStep;
            if (renderState.x > targetDist) renderState.x = targetDist;
        }

        // --- NEW: Smooth HUD Interpolation ---
        const v = state.observation.vehicle;
        renderState.hudSoc = renderState.hudSoc || v.battery_soc_pct;
        renderState.hudTemp = renderState.hudTemp || v.battery_temp_c;
        renderState.hudSolar = renderState.hudSolar || v.solar_power_generated_w;
        renderState.hudSpeed = renderState.hudSpeed || v.current_speed_kph;

        // Smoothly lerp HUD values
        renderState.hudSoc += (v.battery_soc_pct - renderState.hudSoc) * 0.1;
        renderState.hudTemp += (v.battery_temp_c - renderState.hudTemp) * 0.1;
        renderState.hudSolar += (v.solar_power_generated_w - renderState.hudSolar) * 0.1;
        renderState.hudSpeed += (v.current_speed_kph - renderState.hudSpeed) * 0.1;

        // Update DOM elements for real-time feel
        document.getElementById('tel-speed').innerText = `${renderState.hudSpeed.toFixed(1)} kph`;
        
        const socEl = document.getElementById('tel-soc');
        socEl.innerText = `${renderState.hudSoc.toFixed(1)}%`;
        socEl.className = `val ${renderState.hudSoc < 30 ? 'crit' : ''}`;
        
        const tempEl = document.getElementById('tel-temp');
        tempEl.innerText = `${renderState.hudTemp.toFixed(1)} °C`;
        tempEl.className = `val ${renderState.hudTemp > 50 ? 'crit' : ''}`;
        
        document.getElementById('tel-solar').innerText = `${Math.round(renderState.hudSolar)} W`;
        
        document.getElementById('hud-step').innerText = `Step: ${state.observation.waypoint_index} | ${(renderState.x * 1000).toFixed(0)}m / ${(currentTask.total_distance_km*1000).toFixed(0)}m`;

        
        let p1 = getPosAtKm(renderState.x - 0.001);
        let p2 = getPosAtKm(renderState.x + 0.001);
        const carPos = getPosAtKm(renderState.x);
        
        // --- 1. Top Canvas: Side-Scrolling Sensor Window ---
        ctxS.clearRect(0, 0, sensorCanvas.width, sensorCanvas.height);
        
        // Background Grid
        drawGrid(ctxS, sensorCanvas.width, sensorCanvas.height);
        
        const scaleX = 8000; // 1km = 8000px
        const scaleY = 30;   // 1m = 30px (Exaggerated elevation)
        
        ctxS.save();
        // Place car at center-left
        const carScreenX = sensorCanvas.width * 0.3;
        const carScreenY = sensorCanvas.height * 0.6;
        
        ctxS.translate(carScreenX, carScreenY);
        
        // Terrain rendering limits
        let viewDistKmBack = (carScreenX) / scaleX;
        let viewDistKmFwd = (sensorCanvas.width - carScreenX) / scaleX;
        let startKm = Math.max(0, renderState.x - viewDistKmBack - 0.2);
        let endKm = Math.min(totalTrackDist, renderState.x + viewDistKmFwd + 0.2);
        
        let ptsInView = [];
        for(let i=0; i<trackPoints.length; i++) {
            let p = trackPoints[i];
            if (p.km >= startKm && p.km <= endKm) ptsInView.push(p);
        }
        
        if (ptsInView.length > 0) {
            // Draw Ground Fill
            ctxS.beginPath();
            for(let i=0; i<ptsInView.length; i++) {
                let p = ptsInView[i];
                let screenX = (p.km - renderState.x) * scaleX;
                let screenY = -(p.elevation - carPos.elevation) * scaleY;
                if(i===0) ctxS.moveTo(screenX, screenY);
                else ctxS.lineTo(screenX, screenY);
            }
            ctxS.lineTo((ptsInView[ptsInView.length-1].km - renderState.x) * scaleX, sensorCanvas.height);
            ctxS.lineTo((ptsInView[0].km - renderState.x) * scaleX, sensorCanvas.height);
            ctxS.closePath();
            
            let grad = ctxS.createLinearGradient(0, 0, 0, sensorCanvas.height - carScreenY);
            grad.addColorStop(0, '#111827');
            grad.addColorStop(1, '#03060c');
            ctxS.fillStyle = grad;
            ctxS.fill();
            
            // Draw Outer Curbs (Red/White dashes)
            ctxS.lineWidth = 14;
            ctxS.setLineDash([20, 20]);
            ctxS.lineCap = 'round';
            ctxS.beginPath();
            for(let i=0; i<ptsInView.length; i++) {
                let p = ptsInView[i];
                let screenX = (p.km - renderState.x) * scaleX;
                let screenY = -(p.elevation - carPos.elevation) * scaleY;
                if(i===0) ctxS.moveTo(screenX, screenY);
                else ctxS.lineTo(screenX, screenY);
            }
            ctxS.strokeStyle = '#ef4444'; ctxS.stroke();
            
            ctxS.lineDashOffset = 20;
            ctxS.strokeStyle = '#ffffff'; ctxS.stroke();
            ctxS.setLineDash([]);
            
            // Draw Inner Asphalt Track Line
            ctxS.lineWidth = 8;
            ctxS.beginPath();
            for(let i=0; i<ptsInView.length; i++) {
                let p = ptsInView[i];
                let screenX = (p.km - renderState.x) * scaleX;
                let screenY = -(p.elevation - carPos.elevation) * scaleY;
                if(i===0) ctxS.moveTo(screenX, screenY);
                else ctxS.lineTo(screenX, screenY);
            }
            ctxS.strokeStyle = '#1f2937'; ctxS.stroke();
            
            // Waypoint markers
            ctxS.fillStyle = 'rgba(0, 217, 255, 0.5)';
            ctxS.textAlign = 'center';
            ctxS.font = '12px Roboto Mono';
            for(let i=0; i<ptsInView.length; i++) {
                let p = ptsInView[i];
                if (Math.abs(p.km % 1.0) < 0.05) {
                    let screenX = (p.km - renderState.x) * scaleX;
                    let screenY = -(p.elevation - carPos.elevation) * scaleY;
                    ctxS.beginPath();
                    ctxS.moveTo(screenX, screenY);
                    ctxS.lineTo(screenX, screenY - 50);
                    ctxS.strokeStyle = 'rgba(0, 217, 255, 0.5)';
                    ctxS.lineWidth = 2;
                    ctxS.stroke();
                    ctxS.fillText(Math.round(p.km) + 'km', screenX, screenY - 60);
                }
            }
        }
        
        // Draw futuristic off-road car
        let dx = (p2.km - p1.km) * scaleX;
        let dy = -(p2.elevation - p1.elevation) * scaleY;
        let angle = Math.atan2(dy, dx);
        
        let actualSpeed = state.observation.vehicle.current_speed_kph || 0;
        // Add subtle pitch based on speed
        let pitchAngle = (actualSpeed - 60) * 0.002; 
        ctxS.rotate(angle + pitchAngle);
        
        // Hover/Suspension bounce based on speed
        let bounceY = (renderState.speed > 0.001) ? Math.sin(now/50) * 1.5 : 0;
        
        // Chassis shadow
        ctxS.fillStyle = 'rgba(0,0,0,0.5)';
        ctxS.beginPath();
        ctxS.ellipse(0, 5, 45, 10, 0, 0, Math.PI*2);
        ctxS.fill();
        
        // Dirt Particles
        updateAndDrawParticles(ctxS, actualSpeed);
        
        ctxS.translate(0, bounceY);
        
        // Main Body
        ctxS.fillStyle = '#0f172a'; // dark metal
        ctxS.strokeStyle = '#00d9ff'; // neon accent
        ctxS.lineWidth = 2;
        ctxS.beginPath();
        ctxS.roundRect(-45, -25, 90, 20, 8);
        ctxS.fill(); ctxS.stroke();
        
        // Cockpit / Canopy
        ctxS.fillStyle = 'rgba(0, 217, 255, 0.2)';
        ctxS.beginPath();
        ctxS.roundRect(-15, -45, 45, 20, [10, 15, 0, 0]);
        ctxS.fill(); ctxS.stroke();
        
        // Solar Panel on top
        ctxS.fillStyle = '#f59e0b';
        ctxS.fillRect(-10, -47, 35, 4);
        
        // Glowing thruster / exhaust
        if (actualSpeed > 10) {
            ctxS.fillStyle = '#ef4444';
            ctxS.beginPath();
            ctxS.arc(-45, -15, 4 + Math.random()*3, 0, Math.PI*2);
            ctxS.fill();
            // Glow
            ctxS.shadowBlur = 10;
            ctxS.shadowColor = '#ef4444';
            ctxS.fill();
            ctxS.shadowBlur = 0;
        }
        
        // Wheels (Sci-fi hollow design)
        ctxS.strokeStyle = '#10b981';
        ctxS.lineWidth = 4;
        
        // Spinning wheels based on distance
        let wheelRotation = (renderState.x * 1000) % (Math.PI * 2); 

        // Back Wheel
        ctxS.save();
        ctxS.translate(-30, -5 - bounceY);
        ctxS.rotate(wheelRotation);
        ctxS.beginPath();
        ctxS.arc(0, 0, 14, 0, Math.PI*2);
        ctxS.fillStyle = '#000'; ctxS.fill(); ctxS.stroke();
        // Inner rim spinning
        ctxS.beginPath();
        ctxS.moveTo(0, 0);
        ctxS.lineTo(14, 0);
        ctxS.stroke();
        ctxS.restore();
        
        // Front Wheel
        ctxS.save();
        ctxS.translate(30, -5 - bounceY);
        ctxS.rotate(wheelRotation);
        ctxS.beginPath();
        ctxS.arc(0, 0, 14, 0, Math.PI*2);
        ctxS.fillStyle = '#000'; ctxS.fill(); ctxS.stroke();
        // Inner rim spinning
        ctxS.beginPath();
        ctxS.moveTo(0, 0);
        ctxS.lineTo(14, 0);
        ctxS.stroke();
        ctxS.restore();
        
        // Suspension lines
        ctxS.strokeStyle = '#475569';
        ctxS.lineWidth = 2;
        ctxS.beginPath();
        ctxS.moveTo(-30, -25); ctxS.lineTo(-30, -5 - bounceY);
        ctxS.moveTo(30, -25); ctxS.lineTo(30, -5 - bounceY);
        ctxS.stroke();
        
        ctxS.restore();
        
        // --- 2. Bottom Canvas: Global Circuit (Top-Down Mini-map) ---
        ctxC.clearRect(0, 0, circuitCanvas.width, circuitCanvas.height);
        drawGrid(ctxC, circuitCanvas.width, circuitCanvas.height);
        ctxC.save();
        
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        trackPoints.forEach(p => {
            if(p.x < minX) minX = p.x; if(p.x > maxX) maxX = p.x;
            if(p.y < minY) minY = p.y; if(p.y > maxY) maxY = p.y;
        });
        const trackW = maxX - minX;
        const trackH = maxY - minY;
        const scale = Math.min(circuitCanvas.width / (trackW + 100), circuitCanvas.height / (trackH + 100));
        
        ctxC.translate(circuitCanvas.width/2, circuitCanvas.height/2);
        ctxC.scale(scale, scale);
        ctxC.translate(-(minX + trackW/2), -(minY + trackH/2));
        
        drawTrackCircuit(ctxC);
        
        // Blip for car
        ctxC.beginPath();
        ctxC.arc(carPos.x, carPos.y, 10 / scale, 0, Math.PI*2);
        ctxC.fillStyle = '#f59e0b';
        ctxC.fill();
        
        ctxC.beginPath();
        ctxC.arc(carPos.x, carPos.y, 25 / scale, 0, Math.PI*2);
        ctxC.strokeStyle = 'rgba(245, 158, 11, 0.8)';
        ctxC.lineWidth = 3 / scale;
        ctxC.stroke();
        
        ctxC.restore();
    }
    
    requestAnimationFrame(gameLoop);
}

// Events
btnReset.addEventListener('click', resetEpisode);
btnStep.addEventListener('click', () => takeStep(getTaskAwareAction()));
taskSelect.addEventListener('change', resetEpisode);

// Start
init();
