// Taxi-Out Dispatch Analytics Application Logic

let dashboardData = null;
let charts = {};

// Airport and carrier base offsets for simulator
const airportOffsets = {
  'DFW': 17.5,
  'ORD': 19.0,
  'ATL': 16.0,
  'DEN': 16.8,
  'LAX': 17.2,
  'JFK': 22.0,
  'CLT': 18.0
};

const carrierOffsets = {
  'AA': 1.5,
  'DL': 0.8,
  'UA': 1.0,
  'WN': -0.5,
  'B6': 1.2,
  'NK': -0.8
};

async function initDashboard() {
  try {
    const res = await fetch('evaluation_data.json');
    dashboardData = await res.json();
    console.log("Evaluation data loaded successfully", dashboardData);
    
    // Update KPI badges
    if (dashboardData.metrics && dashboardData.metrics.Random_Forest_Regressor) {
      const rfMetrics = dashboardData.metrics.Random_Forest_Regressor.test;
      document.getElementById('kpi-rf-mae').textContent = rfMetrics.MAE_minutes.toFixed(2);
      document.getElementById('kpi-rf-p5').textContent = rfMetrics.Pct_Within_5_Min.toFixed(1);
    }

    initTabs();
    initCharts();
    populateFlightTable(dashboardData.sample_flights);
    initSimulator();
    initSearchFilter();

  } catch (err) {
    console.error("Could not load evaluation_data.json. Using fallback visualizer.", err);
    // Render with mock fallback if opened via file:// protocol
    initTabs();
    initSimulator();
  }
}

function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      tab.classList.add('active');
      const targetId = tab.dataset.tab;
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }
      
      // Resize charts on tab visibility change
      Object.values(charts).forEach(ch => {
        if (ch) ch.resize();
      });
    });
  });
}

function initCharts() {
  if (!dashboardData) return;

  // Chart defaults for modern dark mode
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';

  // 1. Scatter Chart: Actual vs Predicted
  const scatterCtx = document.getElementById('scatterChart');
  if (scatterCtx && dashboardData.sample_flights) {
    const points = dashboardData.sample_flights.map(f => ({
      x: f.TAXI_OUT,
      y: f.PRED_TAXI_OUT_RF
    }));

    charts.scatter = new Chart(scatterCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Predicted vs. Actual',
            data: points,
            backgroundColor: 'rgba(56, 189, 248, 0.45)',
            borderColor: 'rgba(56, 189, 248, 0.8)',
            pointRadius: 4,
            pointHoverRadius: 6
          },
          {
            label: 'Perfect 1:1 Parity',
            data: [{x: 0, y: 0}, {x: 80, y: 80}],
            type: 'line',
            borderColor: 'rgba(244, 63, 94, 0.8)',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              label: (ctx) => `Actual: ${ctx.parsed.x} min | Predicted: ${ctx.parsed.y} min`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Actual Taxi-Out (Minutes)' },
            min: 5,
            max: 75
          },
          y: {
            title: { display: true, text: 'Random Forest Predicted Taxi-Out (Minutes)' },
            min: 5,
            max: 75
          }
        }
      }
    });
  }

  // 2. Congestion Chart: Congestion vs Taxi Time
  const congCtx = document.getElementById('congestionChart');
  if (congCtx && dashboardData.sample_flights) {
    // Bin sample flights by congestion
    const congBins = {};
    dashboardData.sample_flights.forEach(f => {
      const c = Math.round(f.CONGESTION_1H);
      if (!congBins[c]) congBins[c] = [];
      congBins[c].push(f.TAXI_OUT);
    });

    const sortedBins = Object.keys(congBins).map(Number).sort((a,b) => a-b).filter(c => congBins[c].length >= 3);
    const avgTaxi = sortedBins.map(c => {
      const sum = congBins[c].reduce((a, b) => a + b, 0);
      return (sum / congBins[c].length).toFixed(1);
    });

    charts.congestion = new Chart(congCtx, {
      type: 'line',
      data: {
        labels: sortedBins.map(c => `${c} departures`),
        datasets: [{
          label: 'Average Taxi-Out (min)',
          data: avgTaxi,
          borderColor: '#38bdf8',
          backgroundColor: 'rgba(56, 189, 248, 0.15)',
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointBackgroundColor: '#fff',
          pointRadius: 5
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          x: { title: { display: true, text: 'Rolling Concurrent Departures (Past 60 Min)' } },
          y: { title: { display: true, text: 'Taxi-Out Time (Minutes)' }, min: 10 }
        }
      }
    });
  }

  // 3. Model Comparison Bar Chart
  const modelCtx = document.getElementById('modelBarChart');
  if (modelCtx && dashboardData.metrics) {
    const lr = dashboardData.metrics.Linear_Regression_Baseline.test;
    const rf = dashboardData.metrics.Random_Forest_Regressor.test;

    charts.models = new Chart(modelCtx, {
      type: 'bar',
      data: {
        labels: ['MAE (Min - Lower is Better)', 'RMSE (Min - Lower is Better)', 'Accuracy Within ±5m (%)'],
        datasets: [
          {
            label: 'Baseline Linear Regression',
            data: [lr.MAE_minutes, lr.RMSE_minutes, lr.Pct_Within_5_Min],
            backgroundColor: 'rgba(148, 163, 184, 0.6)',
            borderColor: '#94a3b8',
            borderWidth: 1,
            borderRadius: 6
          },
          {
            label: 'Random Forest Regressor',
            data: [rf.MAE_minutes, rf.RMSE_minutes, rf.Pct_Within_5_Min],
            backgroundColor: 'rgba(56, 189, 248, 0.8)',
            borderColor: '#38bdf8',
            borderWidth: 1,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top' }
        },
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  }

  // 4. Feature Importance Horizontal Bar Chart
  const featCtx = document.getElementById('featureChart');
  if (featCtx && dashboardData.top_features) {
    const labels = dashboardData.top_features.map(f => f.feature);
    const data = dashboardData.top_features.map(f => (f.importance * 100).toFixed(1));

    charts.features = new Chart(featCtx, {
      type: 'bar',
      indexAxis: 'y',
      data: {
        labels: labels,
        datasets: [{
          label: 'Feature Importance Contribution (%)',
          data: data,
          backgroundColor: 'rgba(16, 185, 129, 0.75)',
          borderColor: '#10b981',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { title: { display: true, text: 'Contribution Percentage (%)' }, beginAtZero: true }
        }
      }
    });
  }
}

function initSimulator() {
  const form = document.getElementById('sim-form');
  const delaySlider = document.getElementById('sim-delay');
  const delayVal = document.getElementById('delay-val');
  const congSlider = document.getElementById('sim-congestion');
  const congVal = document.getElementById('congestion-val');

  delaySlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    delayVal.textContent = val >= 0 ? `+${val}m` : `${val}m`;
    updateSimulation();
  });

  congSlider.addEventListener('input', (e) => {
    congVal.textContent = `${e.target.value} fl`;
    updateSimulation();
  });

  ['sim-carrier', 'sim-origin', 'sim-dest', 'sim-time'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateSimulation);
  });

  updateSimulation();
}

function updateSimulation() {
  const carrier = document.getElementById('sim-carrier').value;
  const origin = document.getElementById('sim-origin').value;
  const timeStr = document.getElementById('sim-time').value;
  const delay = parseInt(document.getElementById('sim-delay').value);
  const congestion = parseInt(document.getElementById('sim-congestion').value);

  const [hStr, mStr] = timeStr.split(':');
  const hour = parseInt(hStr);
  const min = parseInt(mStr);

  // Approximate Random Forest surface response function
  const baseAirport = airportOffsets[origin] || 17.0;
  const carrOffset = carrierOffsets[carrier] || 0.0;
  
  // Hour rush curve
  let hourSurge = 0;
  if (hour >= 7 && hour <= 9) hourSurge = 4.2;
  else if (hour >= 16 && hour <= 19) hourSurge = 5.1;
  else if (hour >= 11 && hour <= 14) hourSurge = 2.4;

  // Congestion slope (queuing accumulation)
  const congestionSurge = (congestion - 5) * 0.45;

  let predictedTaxi = baseAirport + carrOffset + hourSurge + congestionSurge + (delay > 15 ? 1.5 : 0.0);
  predictedTaxi = Math.max(8.0, Math.min(65.0, predictedTaxi));

  // Compute times
  const actualDepMins = (hour * 60 + min + delay + 1440) % 1440;
  const depTimeFormatted = `${String(Math.floor(actualDepMins / 60)).padStart(2, '0')}:${String(actualDepMins % 60).padStart(2, '0')}`;
  
  const wheelsOffMins = (actualDepMins + Math.round(predictedTaxi)) % 1440;
  const wheelsOffFormatted = `${String(Math.floor(wheelsOffMins / 60)).padStart(2, '0')}:${String(wheelsOffMins % 60).padStart(2, '0')}`;

  // Update DOM
  document.getElementById('sim-pred-out').innerHTML = `${predictedTaxi.toFixed(1)} <span style="font-size: 1.2rem; font-weight: 500;">min</span>`;
  document.getElementById('sim-est-dep').textContent = depTimeFormatted;
  document.getElementById('sim-est-wo').textContent = wheelsOffFormatted;
}

function populateFlightTable(flights) {
  const tbody = document.getElementById('flight-table-body');
  if (!tbody || !flights) return;

  tbody.innerHTML = '';
  // Show first 40 flights
  flights.slice(0, 40).forEach(f => {
    const tr = document.createElement('tr');
    const absErr = Math.abs(f.ERROR_RF);

    let badgeClass = 'badge-moderate';
    let badgeText = `±${absErr.toFixed(1)}m`;
    if (absErr <= 2.0) {
      badgeClass = 'badge-exact';
      badgeText = 'High Precision (≤2m)';
    } else if (absErr <= 5.0) {
      badgeClass = 'badge-close';
      badgeText = 'Within Spec (≤5m)';
    }

    const schedH = String(Math.floor(f.CRS_DEP_TIME / 100)).padStart(2, '0');
    const schedM = String(f.CRS_DEP_TIME % 100).padStart(2, '0');
    const actH = String(Math.floor(f.DEP_TIME / 100)).padStart(2, '0');
    const actM = String(f.DEP_TIME % 100).padStart(2, '0');

    tr.innerHTML = `
      <td>${f.FL_DATE}</td>
      <td style="font-weight: 700; color: #fff;">${f.OP_UNIQUE_CARRIER}</td>
      <td><span style="color: var(--accent-cyan); font-weight: 600;">${f.ORIGIN}</span></td>
      <td>${f.DEST}</td>
      <td>${schedH}:${schedM}</td>
      <td>${actH}:${actM}</td>
      <td>${Math.round(f.CONGESTION_1H)} flights</td>
      <td style="font-weight: 600;">${f.TAXI_OUT.toFixed(1)}m</td>
      <td style="font-weight: 700; color: var(--accent-cyan);">${f.PRED_TAXI_OUT_RF.toFixed(1)}m</td>
      <td style="color: ${f.ERROR_RF >= 0 ? '#10b981' : '#f59e0b'};">${f.ERROR_RF > 0 ? '+' : ''}${f.ERROR_RF.toFixed(1)}m</td>
      <td><span class="status-badge ${badgeClass}">${badgeText}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function initSearchFilter() {
  const searchInput = document.getElementById('flightSearch');
  if (!searchInput) return;

  searchInput.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const rows = document.querySelectorAll('#flight-table-body tr');
    rows.forEach(row => {
      const text = row.innerText.toLowerCase();
      row.style.display = text.includes(term) ? '' : 'none';
    });
  });
}

window.addEventListener('DOMContentLoaded', initDashboard);
