import React, { useState, useEffect } from 'react';
import { 
  Search, 
  ShieldCheck, 
  Activity, 
  Zap, 
  Info, 
  AlertCircle, 
  ChevronRight,
  Menu,
  X,
  ExternalLink,
  MessageSquare
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Nav = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-4">
    <div className="max-w-7xl mx-auto glass rounded-2xl px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: 'var(--sakura-red)' }}>
          <ShieldCheck className="text-white w-5 h-5" />
        </div>
        <span className="font-bold text-xl tracking-tight">NewsGuard</span>
      </div>
    </div>
  </nav>
);

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="glass glass-hover p-6 rounded-2xl flex flex-col gap-4">
    <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center border border-white/10">
      <Icon style={{ color: 'var(--sakura-red)' }} className="w-6 h-6" />
    </div>
    <div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-slate-400 text-sm leading-relaxed">{description}</p>
    </div>
  </div>
);

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeNews = async () => {
    if (!text.trim()) {
      setError('Please enter some news text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      
      if (!res.ok) throw new Error('Prediction failed');
      
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError('An error occurred while analyzing the text. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-20 px-6">
      <Nav />

      <div className="max-w-5xl mx-auto space-y-24">
        {/* Hero Section */}
        <section className="text-center space-y-6">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-block"
          >
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4">
              Fake News <span style={{ color: 'var(--sakura-red)' }}>Detector</span>
            </h1>
            <p className="text-xl md:text-2xl text-slate-400 font-light max-w-2xl mx-auto">
              A Machine Learning powered engine for <span className="text-white">detecting misinformation</span>
            </p>
          </motion.div>

          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-slate-500 max-w-xl mx-auto text-sm md:text-base leading-relaxed"
          >
            Built with advanced NLP techniques and verified by LLM intelligence to ensure high accuracy and provide transparent confidence scoring for every news article.
          </motion.p>

          <div className="flex flex-wrap justify-center gap-4 pt-4">
            <button 
              onClick={() => document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' })}
              className="button-primary px-8 py-3 flex items-center gap-2"
            >
              Start Analyzing <ChevronRight className="w-4 h-4" />
            </button>
            <button className="button-secondary px-8 py-3">Learn More</button>
          </div>
        </section>

        {/* Predictor Section */}
        <section id="analyze" className="scroll-mt-32">
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass p-8 rounded-3xl space-y-6"
          >
            <div className="flex items-center gap-3 border-b border-white/10 pb-6">
              <MessageSquare style={{ color: 'var(--sakura-red)' }} className="w-6 h-6" />
              <h2 className="text-2xl font-bold">Inference Engine</h2>
            </div>

            <div className="space-y-4">
              <textarea 
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste your news article or text here for deep probabilistic analysis..."
                className="w-100 min-h-[200px] bg-black/40 border border-white/10 rounded-2xl p-6 text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-pink-500/50 transition-all resize-none"
                style={{ width: '100%' }}
              />
              
              <div className="flex justify-between items-center">
                <p className="text-xs text-slate-600 italic">
                  Press Ctrl+Enter to analyze
                </p>
                <button 
                  onClick={analyzeNews}
                  disabled={loading}
                  className="button-primary disabled:opacity-50 flex items-center gap-2"
                >
                  {loading ? (
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Activity className="w-4 h-4" />
                  )}
                  {loading ? 'Analyzing...' : 'Execute Inference'}
                </button>
              </div>
            </div>

            <AnimatePresence mode="wait">
              {error && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl flex items-center gap-3 text-sm"
                >
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  {error}
                </motion.div>
              )}

              {result && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="pt-6 border-t border-white/10 space-y-6"
                >
                  <div className="flex flex-col md:flex-row gap-6 items-center justify-between">
                    <div className="space-y-1 text-center md:text-left">
                      <p className="text-sm text-slate-500 uppercase tracking-widest font-bold">Prediction Result</p>
                      <h3 className={`text-4xl font-black ${
                        result.label === 'REAL' ? 'text-emerald-400' : 
                        result.label === 'FAKE' ? 'text-red-400' : 'text-amber-400'
                      }`}>
                        {result.label}
                      </h3>
                    </div>

                    <div className="flex gap-8">
                      <div className="text-center">
                        <p className="text-xs text-slate-600 uppercase font-bold mb-1">Confidence</p>
                        <p className="text-3xl font-bold">{result.confidence}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-xs text-slate-600 uppercase font-bold mb-1">Status</p>
                        <div className="flex items-center gap-2" style={{ color: 'var(--sakura-red)' }}>
                          <ShieldCheck className="w-4 h-4" />
                          {result.llm_verified ? 'LLM Verified' : 'ML Inferred'}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full ${
                        result.label === 'REAL' ? 'bg-emerald-500' : 
                        result.label === 'FAKE' ? 'bg-red-500' : 'bg-amber-500'
                      }`}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </section>

        {/* Feature Grid */}
        <section className="space-y-12">
          <div className="text-center">
            <h2 className="text-3xl font-bold mb-4">Core Detection Features</h2>
            <p className="text-slate-500">Built for accuracy, speed, and complete transparency</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <FeatureCard 
              icon={Activity}
              title="ML Classification"
              description="Fast text processing using TF-IDF and scikit-learn for baseline probability."
            />
            <FeatureCard 
              icon={ShieldCheck}
              title="LLM Verification"
              description="Secondary opinion from Groq LLM on low confidence or uncertain results."
            />
            <FeatureCard 
              icon={Zap}
              title="Instant Analysis"
              description="Process full length news articles in milliseconds to get immediate results."
            />
            <FeatureCard 
              icon={Info}
              title="Confidence Scoring"
              description="Get precise confidence intervals with every prediction for complete transparency."
            />
          </div>
        </section>

        <footer className="text-center pt-10">
          <p className="text-xs text-slate-600 max-w-2xl mx-auto leading-relaxed">
            <AlertCircle className="inline-block w-3 h-3 mr-1" />

          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
