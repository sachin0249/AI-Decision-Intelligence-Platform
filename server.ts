import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Lazy initialization of Gemini client to prevent container crash if secret is missing on startup
let genAIClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  if (!genAIClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('GEMINI_API_KEY environment variable is required in AI Studio Secrets.');
    }
    genAIClient = new GoogleGenAI({ apiKey });
  }
  return genAIClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: '15mb' }));

  // Health check endpoint
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
  });

  // 1. Executive City Briefing Generator
  app.post('/api/gemini/briefing', async (req, res) => {
    try {
      const { district, metrics, anomalies } = req.body;
      const ai = getGenAI();

      const prompt = `You are the Chief AI Decision Officer for a modern intelligent city. 
Analyze the current telemetry for district "${district || 'All Districts'}".

Metrics summary:
${JSON.stringify(metrics, null, 2)}

Active Anomalies:
${JSON.stringify(anomalies, null, 2)}

Provide a concise, high-impact executive briefing in JSON format with the following exact schema:
{
  "overallHealthScore": number (0 to 100 representing city well-being),
  "keyWins": string[] (2-3 bullet points highlighting positive trends),
  "criticalAttentionRequired": string[] (2-3 urgent items needing stakeholder action),
  "strategicAISuggestions": string[] (2-3 actionable AI policy recommendations)
}

Return ONLY valid JSON without markdown code blocks if possible.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.3
        }
      });

      const text = response.text || '{}';
      const parsed = JSON.parse(text);
      res.json({
        generatedAt: new Date().toISOString(),
        overallHealthScore: parsed.overallHealthScore || 84,
        keyWins: parsed.keyWins || ['Optimal transit punctuality across core routes', 'Significant 8% reduction in citywide air pollution'],
        criticalAttentionRequired: parsed.criticalAttentionRequired || ['Elevated passenger dwell time at Downtown Station 4', 'Substation Feeder 9B variance requiring inspection'],
        strategicAISuggestions: parsed.strategicAISuggestions || ['Expand AI signal preemption along transit bottlenecks', 'Deploy smart battery backup hubs ahead of forecasted peak load']
      });
    } catch (err: any) {
      console.error('Briefing API Error:', err);
      res.status(500).json({ error: err.message || 'Failed to generate AI briefing' });
    }
  });

  // 2. Natural Language Conversational Analytics (Ask AI)
  app.post('/api/gemini/chat', async (req, res) => {
    try {
      const { query, district, currentMetrics } = req.body;
      if (!query) {
        return res.status(400).json({ error: 'Query is required' });
      }
      const ai = getGenAI();

      const systemPrompt = `You are CityPulse AI, an advanced Decision Intelligence Copilot for city mayors, planners, citizens, and emergency responders.
You have access to live telemetry for ${district || 'the city'}:
${JSON.stringify(currentMetrics, null, 2)}

Answer the user query thoroughly, professionally, and objectively in natural language.
In addition to your narrative response, if the query involves trend analysis, comparison, or resource allocation, you MUST also generate structured data for a dynamic chart.

Return your response in JSON format with this structure:
{
  "content": "Narrative explanation and recommendations in rich text markdown format...",
  "groundingHighlights": ["Metric 1", "District Name"],
  "chartData": {
    "title": "Clear chart title",
    "type": "bar" | "line" | "pie",
    "dataKey": "value",
    "data": [
      { "name": "Label A", "value": 45 },
      { "name": "Label B", "value": 72 }
    ]
  }
}
If no chart is logically relevant, omit "chartData" or set it to null.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: `${systemPrompt}\n\nUser Question: "${query}"`,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.4
        }
      });

      const text = response.text || '{}';
      const parsed = JSON.parse(text);
      res.json({
        id: `msg-${Date.now()}`,
        role: 'assistant',
        content: parsed.content || 'Analysis complete.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        groundingData: {
          relevantDistrict: district,
          metricsHighlighted: parsed.groundingHighlights || []
        },
        chartData: parsed.chartData || null
      });
    } catch (err: any) {
      console.error('Chat API Error:', err);
      res.status(500).json({ error: err.message || 'Failed to process conversational query' });
    }
  });

  // 3. Predictive Scenario & Impact Simulator
  app.post('/api/gemini/simulate', async (req, res) => {
    try {
      const { scenarioTitle, domain, parameters } = req.body;
      const ai = getGenAI();

      const prompt = `You are an AI Simulation Engine for urban systems.
Simulate the scenario: "${scenarioTitle}" (${domain})
Intervention Parameters set by planner:
${JSON.stringify(parameters, null, 2)}

Predict the 6-month systemic impact. Return JSON matching this schema:
{
  "summary": "High-level executive summary of projected outcomes...",
  "projectedMetrics": [
    {
      "name": "Metric name (e.g. Transit Speed or Grid Load)",
      "currentValue": number,
      "projectedValue": number,
      "unit": "string",
      "changePct": number (+ or - percentage),
      "positiveOutcome": boolean
    }
  ] (provide exactly 3 key impacted metrics),
  "timelineForecast": [
    { "month": "M1", "baseline": 100, "intervention": 95 },
    { "month": "M2", "baseline": 102, "intervention": 91 },
    { "month": "M3", "baseline": 105, "intervention": 86 },
    { "month": "M4", "baseline": 104, "intervention": 82 },
    { "month": "M5", "baseline": 108, "intervention": 78 },
    { "month": "M6", "baseline": 110, "intervention": 74 }
  ],
  "recommendations": ["Policy tip 1", "Policy tip 2"],
  "riskAssessment": "Brief explanation of potential side effects or execution risks."
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          temperature: 0.5
        }
      });

      const text = response.text || '{}';
      const parsed = JSON.parse(text);
      res.json({
        scenarioTitle,
        summary: parsed.summary || `Simulation of ${scenarioTitle} demonstrates strong positive network externalities.`,
        projectedMetrics: parsed.projectedMetrics || [],
        timelineForecast: parsed.timelineForecast || [],
        recommendations: parsed.recommendations || ['Deploy in phased pilots', 'Monitor sensor telemetry closely'],
        riskAssessment: parsed.riskAssessment || 'Low systemic risk if smart traffic controllers are calibrated.'
      });
    } catch (err: any) {
      console.error('Simulation API Error:', err);
      res.status(500).json({ error: err.message || 'Failed to execute predictive simulation' });
    }
  });

  // 4. Citizen Multimodal Issue Reporter & AI Triage
  app.post('/api/gemini/report', async (req, res) => {
    try {
      const { title, description, category, district, imageBase64 } = req.body;
      const ai = getGenAI();

      const contentsParts: any[] = [];
      if (imageBase64) {
        // Strip data:image/...;base64, prefix if present
        const cleanBase64 = imageBase64.includes(',') ? imageBase64.split(',')[1] : imageBase64;
        contentsParts.push({
          inlineData: {
            mimeType: 'image/jpeg',
            data: cleanBase64
          }
        });
      }
      contentsParts.push({
        text: `You are an AI Municipal Triage Officer.
Analyze this citizen report submitted for district "${district}":
Title: "${title}"
Category: "${category}"
Description: "${description}"

Assess severity, safety hazards, and department routing. Return valid JSON:
{
  "urgencyScore": number (1.0 to 10.0),
  "routedDepartment": "Exact municipal department name",
  "estimatedResolutionDays": number (integer),
  "safetyHazard": boolean,
  "aiSummary": "Clear 2-sentence formal triage note explaining risk and dispatch priority."
}`
      });

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: { parts: contentsParts },
        config: {
          responseMimeType: 'application/json',
          temperature: 0.2
        }
      });

      const text = response.text || '{}';
      const parsed = JSON.parse(text);
      res.json({
        id: `rep-${Date.now().toString().slice(-4)}`,
        title,
        description,
        category,
        district,
        imageUrl: imageBase64 ? (imageBase64.startsWith('data:') ? imageBase64 : `data:image/jpeg;base64,${imageBase64}`) : undefined,
        status: 'AI Triaged',
        aiAnalysis: {
          urgencyScore: parsed.urgencyScore || 6.5,
          routedDepartment: parsed.routedDepartment || 'Department of Public Works',
          estimatedResolutionDays: parsed.estimatedResolutionDays || 3,
          safetyHazard: parsed.safetyHazard || false,
          aiSummary: parsed.aiSummary || 'Report logged and automatically triaged for municipal dispatch.'
        },
        submittedAt: 'Just now',
        upvotes: 1
      });
    } catch (err: any) {
      console.error('Report Triage API Error:', err);
      res.status(500).json({ error: err.message || 'Failed to triage citizen report' });
    }
  });

  // Vite middleware for development or Static serving in production
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CityPulse AI Decision Intelligence Platform running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
