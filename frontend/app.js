//GLOBAL STATE 
const appState = {
  role: "user",
  predictions: JSON.parse(localStorage.getItem("predictions")) || [],
  currentPage: "predict",
  currentFilter: "all",
  searchQuery: "",
  lastPrediction: null,
  lastFormData: null,
  charts: {},
  modelInfo: {
    algorithm: "Random Forest",
    accuracy: 0.7597,
    version: "1.0.0",
    last_prediction: null
  }
};

//TOAST NOTIFICATIONS
const Toast = {
  show: (message, type = "info", duration = 3000) => {
    const toast = document.createElement("div");
    toast.className = `toast ${type} fade-in`;
    const icons = {
      success: "check-circle",
      error: "alert-circle",
      info: "info"
    };
    const colors = {
      success: "text-green-500",
      error: "text-red-500",
      info: "text-blue-500"
    };

    toast.innerHTML = `
      <div class="flex items-center gap-3">
        <i data-lucide="${icons[type]}" class="w-5 h-5 ${colors[type]}"></i>
        <span>${message}</span>
      </div>
    `;
    document.body.appendChild(toast);
    lucide.createIcons();
    setTimeout(() => toast.remove(), duration);
  }
};

//  API SERVICE 
const API = {
  baseURL: "http://127.0.0.1:8000",

  predict: async (formData) => {
    try {
      const response = await fetch(`${API.baseURL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });

      if (!response.ok) throw new Error("API request failed");
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  getModelInfo: async () => {
    try {
      const response = await fetch(`${API.baseURL}/model-info`);
      if (!response.ok) throw new Error("Failed to fetch model info");
      const data = await response.json();
      return { success: true, data: data.model_info };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
};

const getDisplayPredictions = () => {
  const records = [...appState.predictions];
  if (appState.lastPrediction && appState.lastFormData) {
    const preview = {
      id: "session-preview",
      prediction: appState.lastPrediction.prediction,
      probability: appState.lastPrediction.probability,
      risk: appState.lastPrediction.risk,
      formData: appState.lastFormData,
      timestamp: new Date().toISOString()
    };
    return [preview, ...records];
  }
  return records;
};

//  VALIDATION 
const Validator = {
  rules: {
    Pregnancies: { min: 0, max: 17, label: "Pregnancies" },
    Glucose: { min: 44, max: 199, label: "Glucose (mg/dL)" },
    BloodPressure: { min: 24, max: 122, label: "Blood Pressure (mmHg)" },
    SkinThickness: { min: 7, max: 99, label: "Skin Thickness (mm)" },
    Insulin: { min: 14, max: 846, label: "Insulin (mu U/ml)" },
    BMI: { min: 18.2, max: 67.1, label: "BMI (kg/m2)" },
    DiabetesPedigreeFunction: {
      min: 0.078,
      max: 2.42,
      label: "Diabetes Pedigree Function"
    },
    Age: { min: 21, max: 81, label: "Age (years)" }
  },

  validate: (formData) => {
    const errors = {};
    for (const [key, value] of Object.entries(formData)) {
      const rule = Validator.rules[key];
      if (!rule) continue;

      const num = parseFloat(value);
      if (isNaN(num) || value === "") {
        errors[key] = `${rule.label} is required`;
      } else if (num < rule.min || num > rule.max) {
        errors[key] = `${rule.label} must be between ${rule.min} and ${rule.max}`;
      }
    }
    return { isValid: Object.keys(errors).length === 0, errors };
  }
};

//  PREDICTION SERVICE 
const PredictionService = {
  savePrediction: (result, formData) => {
    const prediction = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      prediction: result.prediction,
      probability: result.probability,
      risk: result.risk,
      formData
    };
    appState.predictions.unshift(prediction);
    localStorage.setItem("predictions", JSON.stringify(appState.predictions));
    return prediction;
  },

  deletePrediction: (id) => {
    appState.predictions = appState.predictions.filter((p) => p.id !== id);
    localStorage.setItem("predictions", JSON.stringify(appState.predictions));
  },

  clearHistory: () => {
    appState.predictions = [];
    localStorage.setItem("predictions", JSON.stringify(appState.predictions));
  },

  exportCSV: () => {
    if (appState.predictions.length === 0) {
      Toast.show("No predictions to export", "info");
      return;
    }

    const headers = [
      "Date",
      "Prediction",
      "Probability",
      "Risk",
      "Age",
      "BMI",
      "Glucose"
    ];
    const rows = appState.predictions.map((p) => [
      new Date(p.timestamp).toLocaleDateString(),
      p.prediction === 1 ? "Diabetic" : "Non-Diabetic",
      (p.probability * 100).toFixed(2) + "%",
      p.risk,
      p.formData.Age,
      p.formData.BMI.toFixed(1),
      p.formData.Glucose
    ]);

    const csv = [headers, ...rows]
      .map((row) => row.map((cell) => `"${cell}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `predictions_${Date.now()}.csv`);
    link.style.visibility = "hidden";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    Toast.show("CSV exported successfully", "success");
  }
};

//  COMPONENTS 
const Components = {
  navbar: () => `
    <nav class="card border-b sticky top-0 z-50 w-full">
      <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
            <i data-lucide="activity" class="w-6 h-6 text-white"></i>
          </div>
          <div>
            <h1 class="text-xl font-bold text-white">DiabetesPred</h1>
            <p class="text-xs text-gray-400">AI-Powered Risk Assessment</p>
          </div>
        </div>

        <div class="flex items-center gap-4">
          <button onclick="switchRole()" class="px-4 py-2 btn-secondary rounded-lg text-sm font-medium transition">
            <i data-lucide="${appState.role === "user" ? "shield" : "user"}" class="w-4 h-4 inline mr-2"></i>
            ${appState.role === "user" ? "Admin View" : "User View"}
          </button>
        </div>
      </div>

      <div class="border-t border-gray-700 px-6 flex gap-8 overflow-x-auto">
        <button onclick="navigateTo('predict')" class="nav-item ${appState.currentPage === "predict" ? "active" : ""} py-3 text-sm font-medium transition text-gray-300 hover:text-white whitespace-nowrap">
          <i data-lucide="activity" class="w-4 h-4 inline mr-2"></i>Predict Risk
        </button>
        <button onclick="navigateTo('history')" class="nav-item ${appState.currentPage === "history" ? "active" : ""} py-3 text-sm font-medium transition text-gray-300 hover:text-white whitespace-nowrap">
          <i data-lucide="history" class="w-4 h-4 inline mr-2"></i>History
        </button>
        ${appState.role === "admin" ? `
          <button onclick="navigateTo('admin')" class="nav-item ${appState.currentPage === "admin" ? "active" : ""} py-3 text-sm font-medium transition text-gray-300 hover:text-white whitespace-nowrap">
            <i data-lucide="bar-chart-3" class="w-4 h-4 inline mr-2"></i>Admin Dashboard
          </button>
        ` : ""}
      </div>
    </nav>
  `,

  predictForm: () => `
    <div class="max-w-4xl mx-auto px-4">
      <div class="card rounded-xl p-8 slide-up">
        <h2 class="text-2xl font-bold text-white mb-2">Patient Risk Assessment</h2>
        <p class="text-gray-400 mb-8">Enter patient health metrics for AI-powered diabetes risk prediction</p>

        <form id="predictionForm" onsubmit="handlePredict(event)">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            ${Object.entries(Validator.rules).map(([field, rule]) => `
              <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">${rule.label}</label>
                <input type="number" name="${field}" placeholder="Enter ${rule.label.toLowerCase()}" step="0.01"
                  class="input-field w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition" />
                <p id="error-${field}" class="text-red-500 text-xs mt-1"></p>
              </div>
            `).join("")}
          </div>

          <button type="submit" class="w-full btn-primary text-white py-3 rounded-lg font-medium transition hover:shadow-lg">
            <i data-lucide="zap" class="w-5 h-5 inline mr-2"></i>Predict Risk
          </button>
        </form>
      </div>

      <div id="resultContainer"></div>
    </div>
  `,

  resultCard: (result, formData) => {
    const isPositive = result.prediction === 1;
    const riskClass = result.risk.toLowerCase().includes("low")
      ? "bg-risk-low"
      : (result.risk.toLowerCase().includes("medium") || result.risk.toLowerCase().includes("moderate"))
        ? "bg-risk-medium"
        : "bg-risk-high";
    const riskColor = result.risk.toLowerCase().includes("low")
      ? "risk-low"
      : (result.risk.toLowerCase().includes("medium") || result.risk.toLowerCase().includes("moderate"))
        ? "risk-medium"
        : "risk-high";

    return `
      <div class="card rounded-xl p-8 mt-8 slide-up border-l-4 ${riskClass}">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <p class="text-gray-400 text-sm mb-2 font-semibold">PREDICTION</p>
            <p class="text-4xl font-bold text-white">${isPositive ? "Diabetic" : "Non-Diabetic"}</p>
            <p class="text-xs text-gray-400 mt-2">${isPositive ? "High Risk" : "Low Risk"}</p>
          </div>

          <div>
            <p class="text-gray-400 text-sm mb-4 font-semibold">RISK LEVEL</p>
            <div class="flex items-end gap-2">
              <div class="flex gap-1">
                ${["Low", "Medium", "High"].map((level) => {
                  const normalizedRisk = result.risk.toLowerCase().includes("low")
                    ? "Low"
                    : (result.risk.toLowerCase().includes("medium") || result.risk.toLowerCase().includes("moderate"))
                      ? "Medium"
                      : "High";
                  return `
                    <div class="w-12 h-16 rounded-lg ${normalizedRisk === level ? "bg-gradient-to-t from-blue-500 to-blue-400" : "bg-gray-700"} transition"></div>
                  `;
                }).join("")}
              </div>
              <p class="text-lg font-bold ${riskColor}">${result.risk}</p>
            </div>
          </div>

          <div>
            <p class="text-gray-400 text-sm mb-4 font-semibold">PROBABILITY</p>
            <div class="flex flex-col items-center justify-center h-32">
              <p class="text-4xl font-bold text-blue-400">${(result.probability * 100).toFixed(1)}%</p>
              <p class="text-xs text-gray-400 mt-2">Confidence score</p>
            </div>
          </div>
        </div>

        <div class="bg-gray-800 rounded-lg p-6 mb-6">
          <p class="text-gray-400 text-sm mb-4 font-semibold">KEY METRICS</p>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div><p class="text-gray-400">Age</p><p class="text-white font-semibold">${formData.Age}</p></div>
            <div><p class="text-gray-400">BMI</p><p class="text-white font-semibold">${formData.BMI.toFixed(1)}</p></div>
            <div><p class="text-gray-400">Glucose</p><p class="text-white font-semibold">${formData.Glucose} mg/dL</p></div>
            <div><p class="text-gray-400">Blood Pressure</p><p class="text-white font-semibold">${formData.BloodPressure} mmHg</p></div>
          </div>
        </div>

        <div class="flex gap-4">
          <button onclick="savePredictionAndRefresh()" class="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-medium transition">
            <i data-lucide="save" class="w-4 h-4 inline mr-2"></i>Save to History
          </button>
          <button onclick="navigateTo('predict')" class="flex-1 btn-secondary hover:bg-gray-600 text-white py-3 rounded-lg font-medium transition">
            <i data-lucide="plus" class="w-4 h-4 inline mr-2"></i>New Assessment
          </button>
        </div>
      </div>
    `;
  },

  predictionHistory: () => {
    const filtered = appState.currentFilter === "all"
      ? appState.predictions
      : appState.currentFilter === "diabetic"
        ? appState.predictions.filter((p) => p.prediction === 1)
        : appState.predictions.filter((p) => p.prediction === 0);

    const searched = filtered.filter((p) =>
      p.risk.toLowerCase().includes(appState.searchQuery.toLowerCase()) ||
      p.formData.Age.toString().includes(appState.searchQuery)
    );

    return `
      <div class="max-w-6xl mx-auto px-4">
        <div class="card rounded-xl p-8 mb-6 slide-up">
          <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
            <div>
              <h2 class="text-2xl font-bold text-white">Prediction History</h2>
              <p class="text-gray-400 text-sm mt-1">${appState.predictions.length} total predictions</p>
            </div>
            <div class="flex gap-2 w-full md:w-auto">
              <button onclick="PredictionService.exportCSV()" class="flex-1 md:flex-initial px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition">
                <i data-lucide="download" class="w-4 h-4 inline mr-2"></i>Export CSV
              </button>
              <button onclick="clearHistoryModal()" class="flex-1 md:flex-initial px-4 py-2 btn-secondary text-white rounded-lg text-sm font-medium transition hover:bg-gray-600">
                <i data-lucide="trash-2" class="w-4 h-4 inline mr-2"></i>Clear All
              </button>
            </div>
          </div>

          <div class="flex flex-col md:flex-row gap-4 mb-6">
            <input type="text" placeholder="Search by risk or age..." id="searchInput" onkeyup="filterHistory()"
              value="${appState.searchQuery}"
              class="input-field flex-1 px-4 py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" />
            <div class="flex gap-2">
              ${["all", "diabetic", "non-diabetic"].map((filter) => `
                <button onclick="setFilter('${filter}')"
                  class="px-4 py-2 rounded-lg text-sm font-medium transition ${appState.currentFilter === filter ? "bg-blue-600 text-white" : "btn-secondary text-gray-300 hover:bg-gray-600"}">
                  ${filter.charAt(0).toUpperCase() + filter.slice(1).replace("-", " ")}
                </button>
              `).join("")}
            </div>
          </div>

          ${searched.length === 0 ? `
            <div class="text-center py-12">
              <i data-lucide="inbox" class="w-12 h-12 text-gray-500 mx-auto mb-4"></i>
              <p class="text-gray-400">No predictions found</p>
            </div>
          ` : `
            <div class="overflow-x-auto">
              <table class="w-full text-left text-sm">
                <thead class="border-b border-gray-700">
                  <tr>
                    <th class="px-4 py-3 font-semibold text-gray-300">Date & Time</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">Prediction</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">Probability</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">Risk</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">Age</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">BMI</th>
                    <th class="px-4 py-3 font-semibold text-gray-300">Action</th>
                  </tr>
                </thead>
                <tbody>
                  ${searched.map((p) => `
                    <tr class="border-b border-gray-700 hover:bg-gray-800 transition">
                      <td class="px-4 py-3 text-sm">${new Date(p.timestamp).toLocaleDateString()} ${new Date(p.timestamp).toLocaleTimeString()}</td>
                      <td class="px-4 py-3">${p.prediction === 1 ? "Diabetic" : "Non-Diabetic"}</td>
                      <td class="px-4 py-3 font-semibold">${(p.probability * 100).toFixed(1)}%</td>
                      <td class="px-4 py-3"><span class="px-3 py-1 rounded-full text-xs font-medium ${p.risk.toLowerCase().includes("low") ? "bg-green-900 text-green-300" : (p.risk.toLowerCase().includes("medium") || p.risk.toLowerCase().includes("moderate")) ? "bg-orange-900 text-orange-300" : "bg-red-900 text-red-300"}">${p.risk}</span></td>
                      <td class="px-4 py-3">${p.formData.Age}</td>
                      <td class="px-4 py-3">${p.formData.BMI.toFixed(1)}</td>
                      <td class="px-4 py-3"><button onclick="deletePrediction(${p.id})" class="text-red-400 hover:text-red-300 transition"><i data-lucide="trash-2" class="w-4 h-4"></i></button></td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          `}
        </div>
      </div>
    `;
  },

  adminDashboard: () => {
    const displayPredictions = getDisplayPredictions();
    const totalPredictions = displayPredictions.length;
    const diabeticCases = displayPredictions.filter((p) => p.prediction === 1).length;
    const avgProbability = totalPredictions > 0
      ? (displayPredictions.reduce((sum, p) => sum + p.probability, 0) / totalPredictions * 100).toFixed(1)
      : 0;
    const latestPrediction = displayPredictions[0] || null;

    return `
      <div class="max-w-7xl mx-auto px-4">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div class="card rounded-xl p-6 slide-up">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-400 text-sm mb-2">Total Predictions</p>
                <p class="text-4xl font-bold text-white">${totalPredictions}</p>
                <p class="text-xs text-blue-400 mt-2">+${Math.floor(Math.max(1, totalPredictions * 0.15))} this month</p>
              </div>
              <i data-lucide="activity" class="w-12 h-12 text-blue-500 opacity-20"></i>
            </div>
          </div>

          <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.1s;">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-400 text-sm mb-2">Diabetic Cases</p>
                <p class="text-4xl font-bold text-red-400">${diabeticCases}</p>
                <p class="text-xs text-gray-400 mt-2">${totalPredictions > 0 ? ((diabeticCases / totalPredictions) * 100).toFixed(1) : 0}% of total</p>
              </div>
              <i data-lucide="alert-circle" class="w-12 h-12 text-red-500 opacity-20"></i>
            </div>
          </div>

          <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.2s;">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-400 text-sm mb-2">Avg Risk Probability</p>
                <p class="text-4xl font-bold text-orange-400">${avgProbability}%</p>
                <p class="text-xs text-gray-400 mt-2">Model accuracy indicator</p>
              </div>
              <i data-lucide="trending-up" class="w-12 h-12 text-orange-500 opacity-20"></i>
            </div>
          </div>

          <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.3s;">
            <div class="flex items-center justify-between">
              <div>
                <p class="text-gray-400 text-sm mb-2">Dataset Used</p>
                <p class="text-4xl font-bold text-green-400">768</p>
                <p class="text-xs text-gray-400 mt-2">Pima Indians dataset</p>
              </div>
              <i data-lucide="database" class="w-12 h-12 text-green-500 opacity-20"></i>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.4s;">
            <h3 class="text-lg font-bold text-white mb-4">Risk Distribution</h3>
            <div class="chart-container">
              <canvas id="riskChart"></canvas>
            </div>
          </div>

          <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.5s;">
            <h3 class="text-lg font-bold text-white mb-4">Prediction Status</h3>
            <div class="chart-container">
              <canvas id="predictionChart"></canvas>
            </div>
          </div>
        </div>

        <div class="card rounded-xl p-6 slide-up" style="animation-delay: 0.6s;">
          <h3 class="text-lg font-bold text-white mb-6">Model Information</h3>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="border-l-2 border-blue-500 pl-4">
              <p class="text-gray-400 text-sm mb-2">Model Version</p>
              <p class="text-xl font-semibold text-white">${appState.modelInfo.version}</p>
              <p class="text-xs text-gray-400 mt-1">Production - ${appState.modelInfo.algorithm}</p>
            </div>
            <div class="border-l-2 border-green-500 pl-4">
              <p class="text-gray-400 text-sm mb-2">Last Prediction</p>
              <p class="text-xl font-semibold text-white">${latestPrediction ? latestPrediction.risk : 'No recent prediction'}</p>
              <p class="text-xs text-gray-400 mt-1">${latestPrediction ? new Date(latestPrediction.timestamp).toLocaleString() : 'N/A'}</p>
            </div>
            <div class="border-l-2 border-purple-500 pl-4">
              <p class="text-gray-400 text-sm mb-2">Algorithm & Accuracy</p>
              <p class="text-xl font-semibold text-white">${appState.modelInfo.algorithm}</p>
              <p class="text-xs text-gray-400 mt-1">${(appState.modelInfo.accuracy * 100).toFixed(2)}% accuracy on test set</p>
            </div>
          </div>
        </div>
      </div>
    `;
  }
};

// PAGE RENDERING 
const render = () => {
  const app = document.getElementById("app");
  let content = Components.navbar();

  content += '<main class="flex-1 overflow-auto py-8">';

  if (appState.currentPage === "predict") {
    content += Components.predictForm();
  } else if (appState.currentPage === "history") {
    content += Components.predictionHistory();
  } else if (appState.currentPage === "admin" && appState.role === "admin") {
    content += Components.adminDashboard();
  }

  content += "</main>";
  app.innerHTML = content;
  lucide.createIcons();

  if (appState.currentPage === "admin" && appState.role === "admin") {
    setTimeout(initCharts, 100);
  }
};

//  CHART INITIALIZATION 
const initCharts = () => {
  const displayPredictions = getDisplayPredictions();
  const riskCounts = {
    low: displayPredictions.filter((p) => p.risk.toLowerCase().includes("low")).length,
    medium: displayPredictions.filter((p) => p.risk.toLowerCase().includes("medium") || p.risk.toLowerCase().includes("moderate")).length,
    high: displayPredictions.filter((p) => p.risk.toLowerCase().includes("high")).length
  };

  const diabeticCases = displayPredictions.filter((p) => p.prediction === 1).length;
  const nonDiabeticCases = displayPredictions.filter((p) => p.prediction === 0).length;

  const riskCtx = document.getElementById("riskChart");
  if (riskCtx) {
    if (appState.charts.risk) appState.charts.risk.destroy();
    appState.charts.risk = new Chart(riskCtx, {
      type: "doughnut",
      data: {
        labels: ["Low Risk", "Medium Risk", "High Risk"],
        datasets: [{
          data: [riskCounts.low, riskCounts.medium, riskCounts.high],
          backgroundColor: ["#22c55e", "#f97316", "#ef4444"],
          borderColor: "#1e293b",
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#e2e8f0", font: { size: 12 } } }
        }
      }
    });
  }

  const predCtx = document.getElementById("predictionChart");
  if (predCtx) {
    if (appState.charts.prediction) appState.charts.prediction.destroy();
    appState.charts.prediction = new Chart(predCtx, {
      type: "bar",
      data: {
        labels: ["Diabetic", "Non-Diabetic"],
        datasets: [{
          label: "Cases",
          data: [diabeticCases, nonDiabeticCases],
          backgroundColor: ["#ef4444", "#22c55e"],
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: "y",
        plugins: {
          legend: { labels: { color: "#e2e8f0" } }
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } }
        }
      }
    });
  }
};

//  EVENT HANDLERS 
const handlePredict = async (e) => {
  e.preventDefault();
  const form = e.target;
  const formData = {};

  for (const [key] of Object.entries(Validator.rules)) {
    formData[key] = form.elements[key].value;
  }

  const validation = Validator.validate(formData);

  Object.keys(Validator.rules).forEach((key) => {
    const errorEl = document.getElementById(`error-${key}`);
    if (errorEl) errorEl.textContent = "";
  });

  if (!validation.isValid) {
    Object.entries(validation.errors).forEach(([key, error]) => {
      const errorEl = document.getElementById(`error-${key}`);
      if (errorEl) errorEl.textContent = error;
    });
    return;
  }

  for (const key in formData) {
    formData[key] = parseFloat(formData[key]);
  }

  const btn = form.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div><span>Predicting...</span>';

  const result = await API.predict(formData);

  const resultContainer = document.getElementById("resultContainer");
  if (result.success) {
    appState.lastPrediction = result.data;
    appState.lastFormData = formData;
    resultContainer.innerHTML = Components.resultCard(result.data, formData);
    lucide.createIcons();
    Toast.show("Prediction successful", "success");
  } else {
    Toast.show("Error: Failed to connect to API. Ensure backend is running.", "error");
  }

  btn.disabled = false;
  btn.innerHTML = '<i data-lucide="zap" class="w-5 h-5 inline mr-2"></i>Predict Risk';
  lucide.createIcons();
};

const savePredictionAndRefresh = () => {
  if (appState.lastPrediction && appState.lastFormData) {
    PredictionService.savePrediction(appState.lastPrediction, appState.lastFormData);
    Toast.show("Prediction saved to history", "success");
    navigateTo("history");
  }
};

const deletePrediction = (id) => {
  PredictionService.deletePrediction(id);
  Toast.show("Prediction deleted", "info");
  render();
};

const clearHistoryModal = () => {
  if (confirm("Are you sure you want to clear all predictions? This action cannot be undone.")) {
    PredictionService.clearHistory();
    Toast.show("History cleared", "info");
    render();
  }
};

const setFilter = (filter) => {
  appState.currentFilter = filter;
  appState.searchQuery = "";
  render();
};

const filterHistory = () => {
  const input = document.getElementById("searchInput");
  appState.searchQuery = input.value;
  render();
};

const navigateTo = (page) => {
  appState.currentPage = page;
  render();
};

const switchRole = () => {
  appState.role = appState.role === "user" ? "admin" : "user";
  appState.currentPage = "predict";
  render();
};

window.PredictionService = PredictionService;
window.handlePredict = handlePredict;
window.savePredictionAndRefresh = savePredictionAndRefresh;
window.deletePrediction = deletePrediction;
window.clearHistoryModal = clearHistoryModal;
window.setFilter = setFilter;
window.filterHistory = filterHistory;
window.navigateTo = navigateTo;
window.switchRole = switchRole;

//  INITIALIZATION 
const initializeApp = async () => {
  // Fetch real model info from backend
  const modelInfoResult = await API.getModelInfo();
  if (modelInfoResult.success) {
    appState.modelInfo = modelInfoResult.data;
  }
  render();
};

initializeApp();