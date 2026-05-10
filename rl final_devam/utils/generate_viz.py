import json
import os
import webbrowser

def generate_dashboard(output_path: str):
    mappo_path = "outputs/mappo/vis_data.json"
    rbc_path = "outputs/rbc/vis_data.json"
    
    data_sets = {}
    
    if os.path.exists(mappo_path):
        with open(mappo_path, "r") as f:
            data_sets["MAPPO"] = json.load(f)
    
    if os.path.exists(rbc_path):
        with open(rbc_path, "r") as f:
            data_sets["RBC"] = json.load(f)

    if not data_sets:
        print("Error: No visualization data found. Run inference first.")
        return

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CityLearn Multi-Agent Visualization</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #080810;
            --card-bg: rgba(255, 255, 255, 0.03);
            --accent-primary: #00f2fe;
            --accent-secondary: #4facfe;
            --text-primary: #ffffff;
            --text-secondary: #8888aa;
            --charge-color: #00e676;
            --discharge-color: #ff5252;
            --idle-color: #5c5c6c;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Outfit', sans-serif; }
        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            background: radial-gradient(circle at 0% 0%, #16213e 0%, #080810 50%);
            min-height: 100vh;
            padding: 2rem;
        }

        .container { max-width: 1400px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 2rem; }
        .title-group h1 { font-size: 2.5rem; letter-spacing: -1px; margin-bottom: 0.5rem; background: linear-gradient(to right, #fff, #8888aa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        .algo-selector { display: flex; gap: 1rem; margin-top: 1rem; }
        .algo-btn { background: var(--card-bg); border: 1px solid rgba(255, 255, 255, 0.1); color: var(--text-secondary); padding: 0.5rem 1.5rem; border-radius: 12px; cursor: pointer; transition: 0.3s; }
        .algo-btn.active { background: var(--accent-primary); color: #000; border-color: var(--accent-primary); font-weight: 600; box-shadow: 0 0 20px rgba(0, 242, 254, 0.3); }

        .dashboard-grid { display: grid; grid-template-columns: 1fr 350px; gap: 2rem; }
        
        .main-panel { display: flex; flex-direction: column; gap: 2rem; }
        .buildings-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; }
        
        .bldg-card { background: var(--card-bg); border-radius: 24px; padding: 1.5rem; border: 1px solid rgba(255, 255, 255, 0.05); position: relative; transition: 0.3s; }
        .bldg-card:hover { border-color: rgba(255, 255, 255, 0.2); transform: translateY(-5px); }
        .bldg-card.active-charge { border-color: var(--charge-color); box-shadow: 0 0 15px rgba(0, 230, 118, 0.1); }
        .bldg-card.active-discharge { border-color: var(--discharge-color); box-shadow: 0 0 15px rgba(255, 82, 82, 0.1); }

        .bldg-icon { width: 100%; height: 100px; display: flex; align-items: flex-end; justify-content: center; gap: 4px; margin-bottom: 1.5rem; position: relative; }
        .bldg-roof { position: absolute; top: 10px; width: 0; height: 0; border-left: 50px solid transparent; border-right: 50px solid transparent; border-bottom: 30px solid rgba(255, 255, 255, 0.05); }
        .window { width: 15px; height: 15px; background: rgba(255, 255, 255, 0.05); border-radius: 2px; }
        .window.lit { background: #ffd700; box-shadow: 0 0 8px #ffd700; }

        .battery-track { width: 100%; height: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; margin: 1rem 0; overflow: hidden; }
        .battery-level { height: 100%; background: linear-gradient(90deg, #00f2fe, #4facfe); transition: 0.5s cubic-bezier(0.4, 0, 0.2, 1); }

        .status-badge { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 20px; background: rgba(255, 255, 255, 0.05); }
        .status-charge { color: var(--charge-color); background: rgba(0, 230, 118, 0.1); }
        .status-discharge { color: var(--discharge-color); background: rgba(255, 82, 82, 0.1); }

        .side-panel { background: var(--card-bg); border-radius: 32px; padding: 2rem; border: 1px solid rgba(255, 255, 255, 0.05); display: flex; flex-direction: column; height: calc(100vh - 15rem); position: sticky; top: 2rem; }
        .panel-title { font-size: 1.2rem; margin-bottom: 1.5rem; color: var(--accent-primary); display: flex; align-items: center; gap: 0.5rem; }
        .log-container { flex-grow: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
        .log-card { background: rgba(255, 255, 255, 0.02); padding: 1.2rem; border-radius: 20px; border-left: 4px solid var(--accent-secondary); animation: slideIn 0.4s ease; }
        .log-meta { font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
        .log-content { font-size: 0.9rem; line-height: 1.5; color: #e0e0f0; }

        .stats-strip { display: flex; gap: 2rem; margin-bottom: 2rem; }
        .stat-item { flex: 1; background: var(--card-bg); padding: 1.2rem; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.05); }
        .stat-val { font-size: 1.8rem; font-weight: 600; margin-top: 0.3rem; color: var(--accent-primary); }

        .controls { position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%); width: 900px; background: rgba(10, 10, 20, 0.8); backdrop-filter: blur(20px); padding: 1rem 2.5rem; border-radius: 100px; border: 1px solid rgba(255, 255, 255, 0.1); display: flex; align-items: center; gap: 2rem; box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 100; }
        #play-btn { width: 50px; height: 50px; border-radius: 50%; border: none; background: var(--accent-primary); color: #000; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; }
        input[type="range"] { flex: 1; accent-color: var(--accent-primary); cursor: pointer; }

        @keyframes slideIn { from { opacity: 0; transform: translateX(30px); } to { opacity: 1; transform: translateX(0); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>CityLearn Intelligence</h1>
                <div class="algo-selector" id="algo-selector">
                    <!-- Buttons injected -->
                </div>
            </div>
            <div style="text-align: right">
                <div id="clock" style="font-size: 2rem; font-weight: 300">00:00</div>
                <div id="day-info" style="color: var(--text-secondary)">Monday, Day 1</div>
            </div>
        </header>

        <div class="stats-strip">
            <div class="stat-item">
                <div style="color: var(--text-secondary)">Electricity Price</div>
                <div class="stat-val" id="stat-price">$0.00</div>
            </div>
            <div class="stat-item">
                <div style="color: var(--text-secondary)">Grid Demand</div>
                <div class="stat-val" id="stat-load">0.0 kW</div>
            </div>
            <div class="stat-item">
                <div style="color: var(--text-secondary)">Outdoor Temperature</div>
                <div class="stat-val" id="stat-temp">22.0°C</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="main-panel">
                <div class="buildings-grid" id="buildings-grid">
                    <!-- Buildings injected -->
                </div>
                
                <div style="background: var(--card-bg); height: 200px; border-radius: 32px; border: 1px solid rgba(255, 255, 255, 0.05); padding: 2rem; display: flex; align-items: center; justify-content: center; color: var(--text-secondary)">
                    Real-time load profile visualization would go here
                </div>
            </div>

            <div class="side-panel">
                <div class="panel-title">🧠 Scenario Analysis</div>
                <div class="log-container" id="log-container">
                    <!-- Logs injected -->
                </div>
            </div>
        </div>

        <div class="controls">
            <button id="play-btn">▶</button>
            <input type="range" id="time-slider" min="0" max="47" value="0">
            <div id="step-label" style="min-width: 100px; font-weight: 600">Step 00 / 48</div>
        </div>
    </div>

    <script>
        const dataSets = DATA_SETS;
        let activeAlgo = Object.keys(dataSets)[0];
        let currentIdx = 0;
        let isPlaying = false;
        let timer;

        const grid = document.getElementById('buildings-grid');
        const selector = document.getElementById('algo-selector');
        const log = document.getElementById('log-container');

        function init() {
            // Build buttons
            Object.keys(dataSets).forEach(name => {
                const btn = document.createElement('button');
                btn.className = `algo-btn ${name === activeAlgo ? 'active' : ''}`;
                btn.innerText = name;
                btn.onclick = () => switchAlgo(name);
                selector.appendChild(btn);
            });

            // Build buildings
            grid.innerHTML = '';
            for(let i=0; i<5; i++) {
                const card = document.createElement('div');
                card.className = 'bldg-card';
                card.id = `b-${i}`;
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem">
                        <span style="font-weight: 600">Bldg ${i+1}</span>
                        <span id="badge-${i}" class="status-badge">Idle</span>
                    </div>
                    <div class="bldg-icon">
                        <div class="bldg-roof"></div>
                        <div class="window" id="w-${i}-0" style="margin-left: 20px"></div>
                        <div class="window" id="w-${i}-1" style="margin-left: 5px"></div>
                        <div class="window" id="w-${i}-2" style="margin-left: 5px"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary)">Battery Storage</div>
                    <div class="battery-track"><div class="battery-level" id="bat-${i}"></div></div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem">
                        <span id="soc-${i}">0.0%</span>
                        <span id="act-${i}" style="color: var(--text-secondary)">0.00</span>
                    </div>
                `;
                grid.appendChild(card);
            }

            renderStep(0);
        }

        function switchAlgo(name) {
            activeAlgo = name;
            document.querySelectorAll('.algo-btn').forEach(b => b.classList.toggle('active', b.innerText === name));
            renderStep(currentIdx);
        }

        function renderStep(idx) {
            const step = dataSets[activeAlgo].steps[idx];
            currentIdx = idx;

            document.getElementById('clock').innerText = `${step.hour.toString().padStart(2, '0')}:00`;
            document.getElementById('stat-price').innerText = `$${step.price.toFixed(3)}`;
            document.getElementById('stat-load').innerText = `${step.district_load.toFixed(1)} kW`;
            document.getElementById('stat-temp').innerText = `${step.temp.toFixed(1)}°C`;
            document.getElementById('step-label').innerText = `Step ${idx.toString().padStart(2, '0')} / ${dataSets[activeAlgo].steps.length-1}`;
            document.getElementById('time-slider').value = idx;

            step.buildings.forEach((b, i) => {
                const card = document.getElementById(`b-${i}`);
                const badge = document.getElementById(`badge-${i}`);
                const bat = document.getElementById(`bat-${i}`);
                const soc = document.getElementById(`soc-${i}`);
                const act = document.getElementById(`act-${i}`);

                const socPct = (b.soc / 40) * 100;
                bat.style.width = `${socPct}%`;
                soc.innerText = `${socPct.toFixed(0)}%`;
                act.innerText = b.action.toFixed(2);

                card.classList.remove('active-charge', 'active-discharge');
                badge.className = 'status-badge';
                
                if(b.action > 0.05) {
                    card.classList.add('active-charge');
                    badge.innerText = 'Charging';
                    badge.classList.add('status-charge');
                    document.getElementById(`w-${i}-1`).classList.add('lit');
                } else if(b.action < -0.05) {
                    card.classList.add('active-discharge');
                    badge.innerText = 'Discharging';
                    badge.classList.add('status-discharge');
                    document.getElementById(`w-${i}-1`).classList.remove('lit');
                } else {
                    badge.innerText = 'Idle';
                    document.getElementById(`w-${i}-1`).classList.remove('lit');
                }
            });

            addLog(step);
        }

        function addLog(step) {
            const entry = document.createElement('div');
            entry.className = 'log-card';
            
            let logic = "";
            if (step.price > 0.25) logic = `High electricity price ($${step.price.toFixed(2)}). The agents are discharging batteries to avoid high grid costs.`;
            else if (step.hour < 7) logic = "Night mode: Prices are low. Agents are recharging batteries to prepare for the peak demand.";
            else if (step.temp > 28) logic = "High heat alert: Cooling demand is peaking. Agents are using stored energy to stabilize the district load.";
            else logic = "Balanced operations: Monitoring solar production and grid stability to optimize storage levels.";

            entry.innerHTML = `
                <div class="log-meta">Time: ${step.hour}:00 | Algo: ${activeAlgo}</div>
                <div class="log-content">${logic}</div>
            `;
            log.prepend(entry);
            if(log.children.length > 8) log.removeChild(log.lastChild);
        }

        document.getElementById('play-btn').onclick = () => {
            isPlaying = !isPlaying;
            document.getElementById('play-btn').innerText = isPlaying ? '⏸' : '▶';
            if(isPlaying) {
                timer = setInterval(() => {
                    currentIdx = (currentIdx + 1) % dataSets[activeAlgo].steps.length;
                    renderStep(currentIdx);
                }, 800);
            } else {
                clearInterval(timer);
            }
        };

        document.getElementById('time-slider').oninput = (e) => {
            renderStep(parseInt(e.target.value));
        };

        init();
    </script>
</body>
</html>
    """
    
    html_content = html_template.replace("DATA_SETS", json.dumps(data_sets))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Standalone Dashboard created: {output_path}")
    webbrowser.open('file://' + os.path.realpath(output_path))

if __name__ == "__main__":
    generate_dashboard("outputs/dashboard.html")
