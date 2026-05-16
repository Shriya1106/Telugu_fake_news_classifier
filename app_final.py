"""
Telugu Fake News Classifier - Final Enhanced Version
====================================================
Beautiful UI with English Translation
"""

import sys
import os

# Fix PyTorch DLL error on Windows
python_dir = os.path.dirname(sys.executable)
dll_paths = [
    python_dir,
    os.path.join(python_dir, 'Library', 'bin'),
    os.path.join(python_dir, 'DLLs'),
]
for path in dll_paths:
    if os.path.exists(path) and path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify
from src.predict import predict
from src.preprocess import find_trigger_words, clean_telugu_text

app = Flask(__name__)

def translate(text):
    """Translate Telugu text to English using googletrans"""
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src='te', dest='en')
        return result.text
    except Exception as e:
        # Fallback to basic word-by-word if translation fails
        print(f"Translation error: {e}")
        TRANS = {
            "ప్రభుత్వం": "government", "రైతులు": "farmers", "రూ.": "Rs.", "ఉచితంగా": "free",
            "వెంటనే": "immediately", "లింక్": "link", "క్లిక్": "click", "చేయండి": "do",
            "రేపు": "tomorrow", "ఉదయం": "morning", "గంటలకు": "at", "ఇస్రో": "ISRO",
            "కొత్త": "new", "ఉపగ్రహాన్ని": "satellite", "ప్రయోగించనుంది": "will launch",
            "ఆకు": "leaf", "రసం": "juice", "తాగితే": "if drink", "జబ్బు": "disease",
            "అయినా": "any", "రోజుల్లో": "in days", "తగ్గిపోతుంది": "will reduce",
            "గ్యారంటీ": "guarantee", "షేర్": "share", "హైదరాబాద్": "Hyderabad",
            "నిన్న": "yesterday", "భారీ": "heavy", "వర్షం": "rain", "కారణంగా": "due to",
            "పలు": "many", "ప్రాంతాలు": "areas", "జలమయం": "flooded", "అయ్యాయి": "became",
            "కరోనా": "corona", "మూడో": "third", "దశ": "phase", "వస్తోంది": "is coming",
            "అందరూ": "everyone", "ఇళ్లలోనే": "at home", "ఉండాలి": "stay", "WHO": "WHO",
            "హెచ్చరిక": "warning", "ఫార్వర్డ్": "forward", "ఏ": "any", "100%": "100%",
            "ఇస్తోంది": "is giving", "10,000": "10,000",
        }
        words = text.split()
        return ' '.join([TRANS.get(w.strip('.,!?;:'), w) for w in words])

HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Telugu Fake News Classifier</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px;position:relative;overflow-x:hidden}
body::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>');opacity:0.3;pointer-events:none}
.container{max-width:1500px;margin:0 auto;background:rgba(255,255,255,0.98);border-radius:32px;padding:50px;box-shadow:0 30px 90px rgba(0,0,0,0.4);backdrop-filter:blur(10px);position:relative;animation:slideUp 0.6s ease-out}
@keyframes slideUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
.header{text-align:center;margin-bottom:40px;position:relative}
.header::after{content:'';display:block;width:100px;height:4px;background:linear-gradient(90deg,#667eea,#764ba2);margin:20px auto;border-radius:2px}
h1{color:#667eea;font-size:3.2rem;margin-bottom:12px;font-weight:800;letter-spacing:-1px;text-shadow:2px 2px 4px rgba(0,0,0,0.1)}
.subtitle{color:#666;font-size:1.2rem;font-weight:500;opacity:0.9}
.badge{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:600;margin-top:10px;box-shadow:0 4px 12px rgba(102,126,234,0.3)}
.content{display:grid;grid-template-columns:1fr 1fr;gap:35px;margin-top:30px}
.section{padding:30px;background:linear-gradient(135deg,#f8f9fa 0%,#ffffff 100%);border-radius:20px;box-shadow:0 8px 24px rgba(0,0,0,0.08);transition:all 0.3s;border:1px solid rgba(102,126,234,0.1)}
.section:hover{box-shadow:0 12px 32px rgba(0,0,0,0.12);transform:translateY(-2px)}
.section h3{color:#333;font-size:1.4rem;margin-bottom:20px;font-weight:700;display:flex;align-items:center;gap:10px}
textarea{width:100%;padding:18px;border:2px solid #e0e0e0;border-radius:14px;font-size:16px;min-height:220px;font-family:inherit;transition:all 0.3s;background:#fff;resize:vertical}
textarea:focus{outline:none;border-color:#667eea;box-shadow:0 0 0 4px rgba(102,126,234,0.15);transform:scale(1.01)}
textarea::placeholder{color:#aaa}
.btn-group{display:flex;gap:12px;margin-top:18px}
button{flex:1;padding:16px;border:none;border-radius:14px;font-size:16px;font-weight:700;cursor:pointer;transition:all 0.3s;text-transform:uppercase;letter-spacing:0.5px}
.btn-primary{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;box-shadow:0 6px 20px rgba(102,126,234,0.4)}
.btn-primary:hover{transform:translateY(-3px);box-shadow:0 10px 30px rgba(102,126,234,0.5)}
.btn-primary:active{transform:translateY(-1px)}
.btn-secondary{background:#f0f0f0;color:#666;box-shadow:0 4px 12px rgba(0,0,0,0.1)}
.btn-secondary:hover{background:#e0e0e0;transform:translateY(-2px)}
.samples-section{margin-top:25px;padding:20px;background:rgba(102,126,234,0.05);border-radius:14px;border:2px dashed rgba(102,126,234,0.2)}
.samples-section h4{margin-bottom:12px;color:#667eea;font-weight:700;font-size:1.1rem}
.sample-btn{width:100%;padding:14px 16px;margin:6px 0;background:#fff;border:2px solid #e0e0e0;border-radius:10px;text-align:left;cursor:pointer;font-size:14px;transition:all 0.3s;font-weight:500;position:relative;overflow:hidden}
.sample-btn::before{content:'';position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(102,126,234,0.1),transparent);transition:left 0.5s}
.sample-btn:hover::before{left:100%}
.sample-btn:hover{border-color:#667eea;background:#f8f9ff;transform:translateX(5px);box-shadow:0 4px 12px rgba(102,126,234,0.2)}
.result{padding:40px;border-radius:20px;margin-bottom:25px;text-align:center;color:#fff;animation:resultPop 0.5s ease-out;box-shadow:0 12px 40px rgba(0,0,0,0.2);position:relative;overflow:hidden}
@keyframes resultPop{from{opacity:0;transform:scale(0.9)}to{opacity:1;transform:scale(1)}}
.result::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(circle,rgba(255,255,255,0.1) 0%,transparent 70%);animation:pulse 3s infinite}
@keyframes pulse{0%,100%{transform:scale(1);opacity:0.5}50%{transform:scale(1.1);opacity:0.8}}
.result.real{background:linear-gradient(135deg,#10b981,#059669);box-shadow:0 12px 40px rgba(16,185,129,0.4)}
.result.fake{background:linear-gradient(135deg,#ef4444,#dc2626);box-shadow:0 12px 40px rgba(239,68,68,0.4)}
.result.unverifiable{background:linear-gradient(135deg,#f59e0b,#d97706);box-shadow:0 12px 40px rgba(245,158,11,0.4)}
.result h2{font-size:3rem;margin:15px 0;font-weight:800;text-shadow:2px 2px 8px rgba(0,0,0,0.2)}
.result .icon{font-size:4rem;animation:bounce 1s ease-out}
@keyframes bounce{0%,100%{transform:scale(1)}50%{transform:scale(1.2)}}
.prob-section{margin-top:25px;padding:25px;background:#fff;border-radius:16px;box-shadow:0 6px 20px rgba(0,0,0,0.08)}
.prob-section h4{margin-bottom:20px;color:#333;font-weight:700;font-size:1.2rem}
.prob-bar{margin:18px 0}
.prob-label{display:flex;justify-content:space-between;margin-bottom:8px;font-size:15px;font-weight:600;color:#333}
.bar{height:35px;background:#e8e8e8;border-radius:10px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.1)}
.bar-fill{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);transition:width 0.8s cubic-bezier(0.4,0,0.2,1);position:relative;box-shadow:0 2px 8px rgba(102,126,234,0.4)}
.bar-fill::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.3),transparent);animation:shimmer 2s infinite}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.keywords{padding:25px;background:#fff;border-radius:16px;margin-top:25px;border:2px solid #fee2e2;box-shadow:0 6px 20px rgba(239,68,68,0.1)}
.keywords strong{color:#dc2626;font-size:1.1rem;display:block;margin-bottom:12px}
.keyword-badge{display:inline-block;padding:8px 16px;margin:6px;background:linear-gradient(135deg,#fee2e2,#fecaca);color:#dc2626;border-radius:10px;font-size:14px;font-weight:700;box-shadow:0 4px 12px rgba(220,38,38,0.2);transition:all 0.3s}
.keyword-badge:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(220,38,38,0.3)}
.translation{padding:25px;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border:2px solid #bae6fd;border-radius:16px;margin-top:25px;box-shadow:0 6px 20px rgba(3,105,161,0.1)}
.translation h4{color:#0369a1;margin-bottom:12px;font-weight:700;font-size:1.1rem;display:flex;align-items:center;gap:8px}
.translation p{color:#075985;line-height:1.8;font-size:15px}
.loading{text-align:center;padding:50px}
.spinner{border:5px solid #f3f3f3;border-top:5px solid #667eea;border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin:25px auto}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.checkbox{display:flex;align-items:center;gap:12px;margin:18px 0;padding:12px;background:rgba(102,126,234,0.05);border-radius:10px;transition:all 0.3s}
.checkbox:hover{background:rgba(102,126,234,0.1)}
.checkbox input{width:22px;height:22px;cursor:pointer;accent-color:#667eea}
.checkbox label{cursor:pointer;font-weight:600;margin:0;color:#333;font-size:15px}
.waiting{text-align:center;padding:80px 20px;color:#999}
.waiting h3{font-size:2rem;margin-bottom:15px;color:#667eea}
.waiting p{font-size:1.1rem}
@media(max-width:768px){.content{grid-template-columns:1fr}h1{font-size:2.2rem}.container{padding:25px}}
</style></head><body>
<div class="container">
<div class="header">
<h1>🛡️ Telugu Fake News Classifier</h1>
<p class="subtitle">AI-Powered Misinformation Detection with English Translation</p>
<span class="badge">✨ Powered by Machine Learning</span>
</div>
<div class="content">
<div class="section">
<h3>📝 Input</h3>
<textarea id="input" placeholder="ఇక్కడ తెలుగు టెక్స్ట్ పేస్ట్ చేయండి… (Paste Telugu text here)"></textarea>
<div class="checkbox"><input type="checkbox" id="showTrans" checked><label for="showTrans">🌐 Show English Translation</label></div>
<div class="btn-group">
<button class="btn-primary" onclick="analyze()">🔍 Analyze</button>
<button class="btn-secondary" onclick="clearAll()">🗑️ Clear</button>
</div>
<div class="samples-section">
<h4>📋 Try Sample Texts:</h4>
<button class="sample-btn" onclick="setSample(0)">🚨 Sample 1: Fake - Government Scam</button>
<button class="sample-btn" onclick="setSample(1)">✅ Sample 2: Real - ISRO News</button>
<button class="sample-btn" onclick="setSample(2)">🚨 Sample 3: Fake - Health Scam</button>
<button class="sample-btn" onclick="setSample(3)">✅ Sample 4: Real - Weather News</button>
<button class="sample-btn" onclick="setSample(4)">⚠️ Sample 5: Unverifiable - WHO Warning</button>
</div>
</div>
<div class="section"><div id="output"><div class="waiting">
<h3>⏳ Waiting for input...</h3><p>Enter Telugu text and click Analyze to detect fake news</p></div></div></div>
</div></div>
<script>
const samples=[
"ప్రభుత్వం రైతులందరికీ రూ. 10,000 ఉచితంగా ఇస్తోంది. వెంటనే ఈ లింక్ క్లిక్ చేయండి.",
"రేపు ఉదయం 10 గంటలకు ఇస్రో కొత్త ఉపగ్రహాన్ని ప్రయోగించనుంది.",
"ఈ ఆకు రసం తాగితే ఏ జబ్బు అయినా 2 రోజుల్లో తగ్గిపోతుంది. 100% గ్యారంటీ! షేర్ చేయండి.",
"హైదరాబాద్‌లో నిన్న భారీ వర్షం కారణంగా పలు ప్రాంతాలు జలమయం అయ్యాయి.",
"కరోనా మూడో దశ వస్తోంది, అందరూ ఇళ్లలోనే ఉండాలి అని WHO హెచ్చరిక. ఫార్వర్డ్ చేయండి."
];
function setSample(i){document.getElementById('input').value=samples[i];document.getElementById('input').focus()}
function clearAll(){document.getElementById('input').value='';document.getElementById('output').innerHTML='<div class="waiting"><h3>⏳ Waiting for input...</h3><p>Enter Telugu text and click Analyze to detect fake news</p></div>'}
async function analyze(){
const text=document.getElementById('input').value.trim();
const showTrans=document.getElementById('showTrans').checked;
if(!text){alert('⚠️ Please enter Telugu text!');return}
document.getElementById('output').innerHTML='<div class="loading"><div class="spinner"></div><p style="font-size:1.1rem;color:#667eea;font-weight:600">Analyzing text... (5-10 seconds)</p></div>';
try{
const res=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,showTrans})});
const data=await res.json();
if(data.error){document.getElementById('output').innerHTML='<div style="padding:30px;background:#fee2e2;color:#dc2626;border-radius:16px;box-shadow:0 8px 24px rgba(220,38,38,0.2)"><h3 style="font-size:1.5rem;margin-bottom:10px">❌ Error</h3><p style="font-size:1.1rem">'+data.error+'</p></div>';return}
const cls=data.label.toLowerCase();
const icon=data.label==='Real'?'✅':data.label==='Fake'?'🚨':'⚠️';
let kw='';
if(data.keywords&&data.keywords.length>0){
kw='<div class="keywords"><strong>⚠️ Suspicious Keywords Detected:</strong><br>'+data.keywords.map(k=>'<span class="keyword-badge">'+k+'</span>').join('')+'</div>';
}else{
kw='<div class="keywords" style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-color:#86efac"><strong style="color:#16a34a;font-size:1.1rem">✅ No Suspicious Keywords Found</strong></div>';
}
let trans='';
if(data.translation){
trans='<div class="translation"><h4>🌐 English Translation:</h4><p>'+data.translation+'</p></div>';
}
document.getElementById('output').innerHTML='<div class="result '+cls+'"><div class="icon">'+icon+'</div><h2>'+data.label+'</h2><p style="font-size:1.3rem;opacity:0.95;font-weight:600">Confidence: '+(data.confidence*100).toFixed(1)+'%</p></div>'+trans+'<div class="prob-section"><h4>📊 Probability Distribution:</h4>'+Object.entries(data.probabilities).map(([l,p])=>'<div class="prob-bar"><div class="prob-label"><span>'+l+'</span><span>'+(p*100).toFixed(1)+'%</span></div><div class="bar"><div class="bar-fill" style="width:'+(p*100)+'%"></div></div></div>').join('')+'</div>'+kw;
}catch(e){
document.getElementById('output').innerHTML='<div style="padding:30px;background:#fee2e2;color:#dc2626;border-radius:16px;box-shadow:0 8px 24px rgba(220,38,38,0.2)"><h3 style="font-size:1.5rem;margin-bottom:10px">❌ Error</h3><p style="font-size:1.1rem">'+e.message+'</p></div>';
}}
</script></body></html>"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        show_trans = data.get('showTrans', False)
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction
        label, confidence, prob_dict = predict(text)
        
        # Get keywords
        cleaned = clean_telugu_text(text)
        keywords = find_trigger_words(cleaned)
        
        # Translation
        translation = translate(text) if show_trans else None
        
        return jsonify({
            'label': label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'keywords': keywords,
            'translation': translation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Telugu Fake News Classifier - Enhanced Version")
    print("=" * 70)
    print("\n✅ Features:")
    print("  - Beautiful modern UI")
    print("  - English translation")
    print("  - 5 sample inputs")
    print("  - Real-time analysis")
    print("\n🌐 Opening at: http://localhost:5000\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
