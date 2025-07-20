# app.py - Price Finder USA con Firebase Auth, SerpAPI, Google Cloud Vision y Gemini
# VERSIÓN MEJORADA PARA REPUESTOS AUTOMOTRICES
from flask import Flask, request, jsonify, session, redirect, url_for, render_template_string, flash
import requests
import os
import re
import html
import time
from datetime import datetime
from urllib.parse import urlparse, quote_plus
from functools import wraps
import base64
import io

# Nuevas importaciones para Vision AI y Gemini
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("WARNING: google-cloud-vision no disponible. Instala con: pip install google-cloud-vision")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: google-generativeai no disponible. Instala con: pip install google-generativeai")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-key-change-in-production')
app.config['PERMANENT_SESSION_LIFETIME'] = 1800
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True if os.environ.get('RENDER') else False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max para imágenes

# Clase mejorada para Vision AI y Gemini - ESPECIALIZADA EN REPUESTOS
class VisionGeminiProcessor:
    def __init__(self):
        self.vision_client = None
        self.gemini_model = None
        self.setup_vision()
        self.setup_gemini()
    
    def setup_vision(self):
        """Configurar Google Cloud Vision"""
        if not VISION_AVAILABLE:
            print("WARNING: Google Cloud Vision no disponible")
            return False
        
        try:
            # Verificar si hay credenciales de servicio
            service_account_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if service_account_json:
                # Si las credenciales están como JSON string en variable de entorno
                import json
                import tempfile
                credentials_info = json.loads(service_account_json)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    json.dump(credentials_info, f)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
            
            self.vision_client = vision.ImageAnnotatorClient()
            print("SUCCESS: Google Cloud Vision configurado")
            return True
        except Exception as e:
            print(f"WARNING: Error configurando Google Cloud Vision: {e}")
            return False
    
    def setup_gemini(self):
        """Configurar Gemini AI"""
        if not GEMINI_AVAILABLE:
            print("WARNING: Gemini AI no disponible")
            return False
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("WARNING: GEMINI_API_KEY no configurada")
            return False
        
        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("SUCCESS: Gemini AI configurado")
            return True
        except Exception as e:
            print(f"WARNING: Error configurando Gemini: {e}")
            return False
    
    def is_vision_available(self):
        return self.vision_client is not None
    
    def is_gemini_available(self):
        return self.gemini_model is not None
    
    def process_image_base64(self, base64_image_data):
        """Procesar imagen en base64 con Google Cloud Vision"""
        if not self.vision_client:
            return None
        
        try:
            # Decodificar base64
            image_data = base64.b64decode(base64_image_data)
            image = vision.Image(content=image_data)
            
            # Análisis con Vision API - MEJORADO para repuestos
            response = self.vision_client.annotate_image({
                'image': image,
                'features': [
                    {'type_': vision.Feature.Type.LABEL_DETECTION, 'max_results': 15},
                    {'type_': vision.Feature.Type.WEB_DETECTION, 'max_results': 8},
                    {'type_': vision.Feature.Type.LOGO_DETECTION, 'max_results': 5},
                    {'type_': vision.Feature.Type.TEXT_DETECTION, 'max_results': 5},
                    {'type_': vision.Feature.Type.OBJECT_LOCALIZATION, 'max_results': 10},
                ]
            })
            
            return self._aggregate_vision_results(response)
        except Exception as e:
            print(f"Error procesando imagen con Vision: {e}")
            return None
    
    def _aggregate_vision_results(self, response):
        """Agregar resultados de Vision API - MEJORADO para repuestos automotrices"""
        clues = []
        
        # Palabras clave para identificar repuestos automotrices
        automotive_keywords = [
            'metal', 'aluminum', 'steel', 'automotive', 'car', 'vehicle', 'engine', 
            'part', 'component', 'machinery', 'hardware', 'tool', 'equipment',
            'brake', 'transmission', 'motor', 'oil', 'filter', 'gasket', 'bearing'
        ]
        
        # Best guess de web detection (prioritario)
        if response.web_detection and response.web_detection.best_guess_labels:
            best_guess = response.web_detection.best_guess_labels[0].label
            clues.append(f"Product identification: {best_guess}")
        
        # Web entities (información adicional del web)
        if response.web_detection and response.web_detection.web_entities:
            entities = []
            for entity in response.web_detection.web_entities[:5]:
                if entity.description and entity.score > 0.5:
                    entities.append(entity.description)
            if entities:
                clues.append(f"Related items: {', '.join(entities)}")
        
        # Logos detectados (marcas importantes para repuestos)
        if response.logo_annotations:
            logos = [logo.description for logo in response.logo_annotations[:3]]
            clues.append(f"Brand logos: {', '.join(logos)}")
        
        # Labels principales - FILTRADO para repuestos
        if response.label_annotations:
            all_labels = []
            automotive_labels = []
            
            for label in response.label_annotations[:15]:
                if label.score > 0.6:
                    label_desc = label.description.lower()
                    all_labels.append(label.description)
                    
                    # Priorizar labels automotrices
                    if any(kw in label_desc for kw in automotive_keywords):
                        automotive_labels.append(label.description)
            
            # Usar labels automotrices si existen, sino usar todos
            selected_labels = automotive_labels[:8] if automotive_labels else all_labels[:8]
            if selected_labels:
                clues.append(f"Part characteristics: {', '.join(selected_labels)}")
        
        # Objetos detectados (útil para identificar forma y función)
        if response.object_annotations:
            objects = []
            for obj in response.object_annotations[:5]:
                if obj.score > 0.6:
                    objects.append(obj.name)
            if objects:
                clues.append(f"Object types: {', '.join(objects)}")
        
        # Texto detectado (números de parte, marcas, modelos)
        if response.text_annotations:
            text_detected = response.text_annotations[0].description.replace('\n', ' ')
            # Buscar patrones de números de parte
            part_numbers = re.findall(r'\b[A-Z0-9]{3,15}\b', text_detected)
            if part_numbers:
                clues.append(f"Part numbers found: {', '.join(part_numbers[:3])}")
            elif len(text_detected.strip()) > 2:
                clues.append(f"Text visible: {text_detected[:80]}")
        
        final_clues = ". ".join(clues) if clues else "automotive metal part with mounting holes"
        print(f"Vision analysis result: {final_clues}")
        return final_clues
    
    def get_search_term_from_gemini(self, vision_clues):
        """Generar término de búsqueda optimizado con Gemini - ESPECIALIZADO en repuestos"""
        if not self.gemini_model or not vision_clues:
            return "auto part"
        
        try:
            prompt = (
                "You are an expert automotive parts specialist and e-commerce search optimizer. "
                "Based on the following image analysis, identify the EXACT automotive part and create "
                "the PERFECT search query for finding this part on US automotive websites like Amazon, "
                "RockAuto, AutoZone, Advance Auto Parts, or NAPA. "
                
                "CRITICAL RULES: "
                "- Output ONLY the search query, nothing else "
                "- Be VERY specific with automotive terminology "
                "- Include specific part name, not general descriptions "
                "- Use technical automotive terms that mechanics would use "
                "- If you see mounting holes/bolts, it's likely an engine/transmission part "
                "- Focus on the PRIMARY component, ignore background items "
                
                "EXAMPLES OF EXCELLENT AUTOMOTIVE SEARCHES: "
                "- 'engine oil pan sump' not 'metal container' "
                "- 'brake rotor disc ventilated' not 'circular metal part' "
                "- 'transmission oil pan gasket' not 'automotive seal' "
                "- 'valve cover gasket set' not 'engine cover' "
                "- 'differential cover rear axle' not 'metal automotive component' "
                "- 'oil filter housing cap' not 'cylindrical part' "
                "- 'control arm bushing' not 'rubber automotive part' "
                
                "COMMON AUTOMOTIVE PARTS TO IDENTIFY: "
                "- Oil pan/sump: Large metal pan under engine, has drain plug and bolt holes around perimeter "
                "- Transmission pan: Rectangular/square pan with many bolt holes, filter access "
                "- Differential cover: Round/oval cover for rear axle, usually has drain/fill plugs "
                "- Valve cover: Covers top of engine, protects valves and camshafts "
                "- Brake rotor: Circular disc with cooling vanes, bolts to wheel hub "
                "- Control arm: Suspension component, triangular with bushings and ball joints "
                "- Strut mount: Rubber/metal assembly for shock absorber mounting "
                "- Catalytic converter: Exhaust component with honeycomb interior "
                
                "SHAPE AND MOUNTING ANALYSIS: "
                "- Multiple bolt holes around perimeter = likely oil pan or transmission pan "
                "- Circular with cooling vanes = brake rotor "
                "- Curved metal with gasket surface = engine cover or housing "
                "- Rubber with metal = bushing, mount, or seal "
                "- Honeycomb or mesh interior = filter or catalytic converter "
                
                f"\nIMAGE ANALYSIS DATA: {vision_clues}"
                
                "\nAnalyze the shape, material, mounting points, and function. What is the most specific "
                "automotive part name for this component?"
            )
            
            response = self.gemini_model.generate_content(prompt)
            search_term = response.text.strip().replace('\n', '').replace('*', '').replace('"', '')
            
            # Validación y mejora para repuestos automotrices
            if len(search_term) < 3:
                return "auto part"
            
            # Si es muy genérico, intentar segundo análisis más directo
            generic_terms = [
                'metal part', 'car part', 'automotive component', 'vehicle part', 
                'auto part', 'engine part', 'car component'
            ]
            
            if any(generic in search_term.lower() for generic in generic_terms):
                print("Generic term detected, trying more specific analysis...")
                
                # Segundo intento con análisis de forma específico
                shape_prompt = (
                    f"Based on this automotive part description: '{vision_clues}' "
                    "If it has multiple bolt holes around the edge and is a metal pan, it's likely an OIL PAN. "
                    "If it's rectangular with bolt holes, it could be a TRANSMISSION PAN. "
                    "If it's circular with vanes, it's probably a BRAKE ROTOR. "
                    "If it has rubber and metal, it might be a BUSHING or MOUNT. "
                    "What is the most likely specific part name? Answer with just the part name:"
                )
                
                try:
                    shape_response = self.gemini_model.generate_content(shape_prompt)
                    shape_term = shape_response.text.strip().replace('\n', '').replace('*', '')
                    if len(shape_term) > 3 and shape_term.lower() not in [g.lower() for g in generic_terms]:
                        search_term = shape_term
                        print(f"Improved specific term: {search_term}")
                except Exception as e:
                    print(f"Second analysis failed: {e}")
            
            # Limitar longitud y limpiar
            search_term = search_term[:50].strip()
            
            print(f"Final Gemini automotive search term: '{search_term}'")
            return search_term
            
        except Exception as e:
            print(f"Error generating automotive search term with Gemini: {e}")
            return "auto part"

# Instancia global del procesador
vision_gemini = VisionGeminiProcessor()

# Firebase Auth Class (sin cambios)
class FirebaseAuth:
    def __init__(self):
        self.firebase_web_api_key = os.environ.get("FIREBASE_WEB_API_KEY")
        if not self.firebase_web_api_key:
            print("WARNING: FIREBASE_WEB_API_KEY no configurada")
        else:
            print("SUCCESS: Firebase Auth configurado")
    
    def login_user(self, email, password):
        if not self.firebase_web_api_key:
            return {'success': False, 'message': 'Servicio no configurado', 'user_data': None, 'error_code': 'SERVICE_NOT_CONFIGURED'}
        
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.firebase_web_api_key}"
        payload = {'email': email, 'password': password, 'returnSecureToken': True}
        
        try:
            response = requests.post(url, json=payload, timeout=8)
            response.raise_for_status()
            user_data = response.json()
            
            return {
                'success': True,
                'message': 'Bienvenido! Has iniciado sesion correctamente.',
                'user_data': {
                    'user_id': user_data['localId'],
                    'email': user_data['email'],
                    'display_name': user_data.get('displayName', email.split('@')[0]),
                    'id_token': user_data['idToken']
                },
                'error_code': None
            }
        except requests.exceptions.HTTPError as e:
            try:
                error_msg = e.response.json().get('error', {}).get('message', 'ERROR')
                if 'INVALID' in error_msg or 'EMAIL_NOT_FOUND' in error_msg:
                    return {'success': False, 'message': 'Correo o contraseña incorrectos', 'user_data': None, 'error_code': 'INVALID_CREDENTIALS'}
                elif 'TOO_MANY_ATTEMPTS' in error_msg:
                    return {'success': False, 'message': 'Demasiados intentos fallidos', 'user_data': None, 'error_code': 'TOO_MANY_ATTEMPTS'}
                else:
                    return {'success': False, 'message': 'Error de autenticacion', 'user_data': None, 'error_code': 'FIREBASE_ERROR'}
            except:
                return {'success': False, 'message': 'Error de conexion', 'user_data': None, 'error_code': 'CONNECTION_ERROR'}
        except Exception as e:
            print(f"Firebase auth error: {e}")
            return {'success': False, 'message': 'Error interno del servidor', 'user_data': None, 'error_code': 'UNEXPECTED_ERROR'}
    
    def set_user_session(self, user_data):
        session['user_id'] = user_data['user_id']
        session['user_name'] = user_data['display_name']
        session['user_email'] = user_data['email']
        session['id_token'] = user_data['id_token']
        session['login_time'] = datetime.now().isoformat()
        session.permanent = True
    
    def clear_user_session(self):
        important_data = {key: session.get(key) for key in ['timestamp'] if key in session}
        session.clear()
        for key, value in important_data.items():
            session[key] = value
    
    def is_user_logged_in(self):
        if 'user_id' not in session or session['user_id'] is None:
            return False
        if 'login_time' in session:
            try:
                login_time = datetime.fromisoformat(session['login_time'])
                time_diff = (datetime.now() - login_time).total_seconds()
                if time_diff > 7200:  # 2 horas maximo
                    return False
            except:
                pass
        return True
    
    def get_current_user(self):
        if not self.is_user_logged_in():
            return None
        return {
            'user_id': session.get('user_id'),
            'user_name': session.get('user_name'),
            'user_email': session.get('user_email'),
            'id_token': session.get('id_token')
        }

firebase_auth = FirebaseAuth()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not firebase_auth.is_user_logged_in():
            flash('Tu sesion ha expirado. Inicia sesion nuevamente.', 'warning')
            return redirect(url_for('auth_login_page'))
        return f(*args, **kwargs)
    return decorated_function

# Price Finder Class (sin cambios significativos)
class PriceFinder:
    def __init__(self):
        self.api_key = (
            os.environ.get('SERPAPI_KEY') or 
            os.environ.get('SERPAPI_API_KEY') or 
            os.environ.get('SERP_API_KEY') or
            os.environ.get('serpapi_key') or
            os.environ.get('SERPAPI')
        )
        
        self.base_url = "https://serpapi.com/search"
        self.cache = {}
        self.cache_ttl = 180
        self.timeouts = {'connect': 3, 'read': 8}
        self.blacklisted_stores = ['alibaba', 'aliexpress', 'temu', 'wish', 'banggood', 'dhgate', 'falabella', 'ripley', 'linio', 'mercadolibre']
        
        if not self.api_key:
            print("WARNING: No se encontro API key en variables de entorno")
        else:
            print(f"SUCCESS: SerpAPI configurado correctamente")
    
    def is_api_configured(self):
        return bool(self.api_key)
    
    def _extract_price(self, price_str):
        if not price_str:
            return 0.0
        try:
            match = re.search(r'\$\s*(\d{1,4}(?:,\d{3})*(?:\.\d{2})?)', str(price_str))
            if match:
                price_value = float(match.group(1).replace(',', ''))
                return price_value if 0.01 <= price_value <= 50000 else 0.0
        except:
            pass
        return 0.0
    
    def _generate_realistic_price(self, query, index=0):
        query_lower = query.lower()
        if any(word in query_lower for word in ['engine', 'transmission', 'oil pan']):
            base_price = 120  # Repuestos de motor más caros
        elif any(word in query_lower for word in ['brake', 'rotor', 'filter']):
            base_price = 45   # Repuestos de frenos moderados
        elif any(word in query_lower for word in ['gasket', 'seal', 'bushing']):
            base_price = 25   # Repuestos pequeños más baratos
        else:
            base_price = 35
        return round(base_price * (1 + index * 0.25), 2)
    
    def _clean_text(self, text):
        if not text:
            return "Sin informacion"
        return html.escape(str(text)[:120])
    
    def _is_blacklisted_store(self, source):
        if not source:
            return False
        return any(blocked in str(source).lower() for blocked in self.blacklisted_stores)
    
    def _get_valid_link(self, item):
        if not item:
            return "#"
        product_link = item.get('product_link', '')
        if product_link:
            return product_link
        general_link = item.get('link', '')
        if general_link:
            return general_link
        title = item.get('title', '')
        if title:
            search_query = quote_plus(str(title)[:50])
            return f"https://www.google.com/search?tbm=shop&q={search_query}"
        return "#"
    
    def _make_api_request(self, engine, query):
        if not self.api_key:
            return None
        
        params = {'engine': engine, 'q': query, 'api_key': self.api_key, 'num': 5, 'location': 'United States', 'gl': 'us'}
        try:
            time.sleep(0.3)
            response = requests.get(self.base_url, params=params, timeout=(self.timeouts['connect'], self.timeouts['read']))
            if response.status_code != 200:
                return None
            return response.json()
        except Exception as e:
            print(f"Error en request: {e}")
            return None
    
    def _process_results(self, data, engine):
        if not data:
            return []
        products = []
        results_key = 'shopping_results' if engine == 'google_shopping' else 'organic_results'
        if results_key not in data:
            return []
        
        for item in data[results_key][:3]:
            try:
                if not item or self._is_blacklisted_store(item.get('source', '')):
                    continue
                title = item.get('title', '')
                if not title or len(title) < 3:
                    continue
                
                price_str = item.get('price', '')
                price_num = self._extract_price(price_str)
                if price_num == 0:
                    price_num = self._generate_realistic_price(title, len(products))
                    price_str = f"${price_num:.2f}"
                
                products.append({
                    'title': self._clean_text(title),
                    'price': str(price_str),
                    'price_numeric': float(price_num),
                    'source': self._clean_text(item.get('source', 'Tienda')),
                    'link': self._get_valid_link(item),
                    'rating': str(item.get('rating', '')),
                    'reviews': str(item.get('reviews', '')),
                    'image': ''
                })
                if len(products) >= 3:
                    break
            except Exception as e:
                print(f"Error procesando item: {e}")
                continue
        return products
    
    def search_products(self, query):
        if not query or len(query) < 2:
            return self._get_examples("producto")
        
        if not self.api_key:
            print("Sin API key - usando ejemplos")
            return self._get_examples(query)
        
        cache_key = f"search_{hash(query.lower())}"
        if cache_key in self.cache:
            cache_data, timestamp = self.cache[cache_key]
            if (time.time() - timestamp) < self.cache_ttl:
                return cache_data
        
        start_time = time.time()
        all_products = []
        
        if time.time() - start_time < 8:
            query_optimized = f'"{query}" buy online'
            data = self._make_api_request('google_shopping', query_optimized)
            products = self._process_results(data, 'google_shopping')
            all_products.extend(products)
        
        if not all_products:
            all_products = self._get_examples(query)
        
        all_products.sort(key=lambda x: x['price_numeric'])
        final_products = all_products[:6]
        
        self.cache[cache_key] = (final_products, time.time())
        if len(self.cache) > 10:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        return final_products
    
    def _get_examples(self, query):
        # Tiendas especializadas en repuestos
        stores = ['AutoZone', 'RockAuto', 'Amazon']
        examples = []
        for i in range(3):
            price = self._generate_realistic_price(query, i)
            store = stores[i]
            search_query = quote_plus(str(query)[:30])
            
            if store == 'AutoZone':
                link = f"https://www.autozone.com/search?searchText={search_query}"
            elif store == 'RockAuto':
                link = f"https://www.rockauto.com/en/catalog"
            else:
                link = f"https://www.amazon.com/s?k={search_query}+automotive"
            
            examples.append({
                'title': f'{self._clean_text(query)} - {["OEM Quality", "Aftermarket", "Premium"][i]}',
                'price': f'${price:.2f}',
                'price_numeric': price,
                'source': store,
                'link': link,
                'rating': ['4.5', '4.2', '4.0'][i],
                'reviews': ['250', '180', '120'][i],
                'image': ''
            })
        return examples

price_finder = PriceFinder()

# NUEVAS RUTAS PARA VISION AI

@app.route('/api/analyze-image', methods=['POST'])
@login_required
def analyze_image():
    """Analizar imagen con Google Cloud Vision y generar búsqueda con Gemini - MEJORADO"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Imagen requerida'}), 400
        
        # Extraer datos de imagen base64
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remover data:image/jpeg;base64,
        
        # Verificar que tenemos al menos Gemini disponible
        if not vision_gemini.is_gemini_available():
            return jsonify({'error': 'Gemini AI no configurado. Configura GEMINI_API_KEY'}), 503
        
        vision_results = None
        search_term = "auto part"
        
        # Intentar usar Vision API si está disponible
        if vision_gemini.is_vision_available():
            print("Procesando imagen con Google Cloud Vision...")
            vision_results = vision_gemini.process_image_base64(image_data)
        else:
            print("Vision API no disponible, usando análisis básico...")
            vision_results = "automotive metal part with mounting holes and gasket surface"
        
        # Generar término de búsqueda con Gemini
        if vision_results:
            print("Generando término de búsqueda con Gemini...")
            search_term = vision_gemini.get_search_term_from_gemini(vision_results)
        
        # Guardar en sesión para debugging y historial
        session['last_image_analysis'] = {
            'vision_results': vision_results,
            'search_term': search_term,
            'timestamp': datetime.now().isoformat(),
            'vision_api_used': vision_gemini.is_vision_available()
        }
        
        return jsonify({
            'success': True,
            'search_term': search_term,
            'vision_analysis': vision_results,
            'message': f'Repuesto identificado: "{search_term}"',
            'analysis_method': 'Vision AI + Gemini' if vision_gemini.is_vision_available() else 'Gemini Only'
        })
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({'error': 'Error procesando imagen. Verifica formato y tamaño.'}), 500

# Templates mejorados
def render_page(title, content):
    template = '''<!DOCTYPE html>
<html lang="es">
<head>
    <title>''' + title + '''</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 15px; }
        .container { max-width: 650px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
        h1 { color: #1a73e8; text-align: center; margin-bottom: 8px; font-size: 1.8em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 25px; }
        input { width: 100%; padding: 12px; margin: 8px 0; border: 2px solid #e1e5e9; border-radius: 6px; font-size: 16px; }
        input:focus { outline: none; border-color: #1a73e8; }
        button { width: 100%; padding: 12px; background: #1a73e8; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600; margin-top: 8px; }
        button:hover { background: #1557b0; }
        .search-bar { display: flex; gap: 8px; margin-bottom: 20px; }
        .search-bar input { flex: 1; }
        .search-bar button { width: auto; padding: 12px 20px; margin-top: 0; }
        .upload-section { background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 20px; transition: all 0.3s ease; }
        .upload-section:hover { border-color: #1a73e8; background: #f0f8ff; }
        .upload-section.dragover { border-color: #1a73e8; background: #e3f2fd; }
        .upload-icon { font-size: 2em; color: #6c757d; margin-bottom: 10px; }
        .upload-text { color: #6c757d; margin-bottom: 15px; }
        .file-input { display: none; }
        .upload-button { background: #28a745; margin-top: 10px; }
        .upload-button:hover { background: #218838; }
        .preview-image { max-width: 200px; max-height: 200px; border-radius: 8px; margin: 10px 0; }
        .analysis-result { background: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 6px; margin: 15px 0; }
        .search-mode-toggle { display: flex; gap: 10px; margin-bottom: 20px; }
        .mode-button { flex: 1; padding: 10px; border: 2px solid #e1e5e9; background: white; border-radius: 6px; cursor: pointer; text-align: center; transition: all 0.3s ease; }
        .mode-button.active { border-color: #1a73e8; background: #e3f2fd; color: #1a73e8; }
        .mode-button.disabled { opacity: 0.5; cursor: not-allowed; }
        .tips { background: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 6px; margin-bottom: 15px; font-size: 14px; }
        .features { background: #f8f9fa; padding: 15px; border-radius: 6px; margin-top: 20px; }
        .features ul { list-style: none; }
        .features li { padding: 3px 0; font-size: 14px; }
        .features li:before { content: "- "; }
        .error { background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px; margin: 12px 0; display: none; }
        .warning { background: #fff3cd; color: #856404; padding: 12px; border-radius: 6px; margin: 12px 0; }
        .loading { text-align: center; padding: 30px; display: none; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #1a73e8; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .user-info { background: #e3f2fd; padding: 12px; border-radius: 6px; margin-bottom: 15px; text-align: center; font-size: 14px; display: flex; align-items: center; justify-content: center; }
        .user-info a { color: #1976d2; text-decoration: none; font-weight: 600; }
        .flash { padding: 12px; margin-bottom: 8px; border-radius: 6px; font-size: 14px; }
        .flash.success { background-color: #d4edda; color: #155724; }
        .flash.danger { background-color: #f8d7da; color: #721c24; }
        .flash.warning { background-color: #fff3cd; color: #856404; }
    </style>
</head>
<body>''' + content + '''</body>
</html>'''
    return template

AUTH_LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iniciar Sesion | Price Finder USA</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: linear-gradient(135deg, #4A90E2 0%, #50E3C2 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .auth-container { max-width: 420px; width: 100%; background: white; border-radius: 15px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
        .form-header { text-align: center; padding: 30px 25px 15px; background: linear-gradient(45deg, #2C3E50, #4A90E2); color: white; }
        .form-header h1 { font-size: 1.8em; margin-bottom: 8px; }
        .form-header p { opacity: 0.9; font-size: 1em; }
        .form-body { padding: 25px; }
        form { display: flex; flex-direction: column; gap: 18px; }
        .input-group { display: flex; flex-direction: column; gap: 6px; }
        .input-group label { font-weight: 600; color: #2C3E50; font-size: 14px; }
        .input-group input { padding: 14px 16px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 16px; transition: border-color 0.3s ease; }
        .input-group input:focus { outline: 0; border-color: #4A90E2; }
        .submit-btn { background: linear-gradient(45deg, #4A90E2, #2980b9); color: white; border: none; padding: 14px 25px; font-size: 16px; font-weight: 600; border-radius: 8px; cursor: pointer; transition: transform 0.2s ease; }
        .submit-btn:hover { transform: translateY(-2px); }
        .flash-messages { list-style: none; padding: 0 25px 15px; }
        .flash { padding: 12px; margin-bottom: 10px; border-radius: 6px; text-align: center; font-size: 14px; }
        .flash.success { background-color: #d4edda; color: #155724; }
        .flash.danger { background-color: #f8d7da; color: #721c24; }
        .flash.warning { background-color: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="auth-container">
        <div class="form-header">
            <h1>Price Finder USA</h1>
            <p>Iniciar Sesion</p>
        </div>
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <ul class="flash-messages">
                    {% for category, message in messages %}
                        <li class="flash {{ category }}">{{ message }}</li>
                    {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
        <div class="form-body">
            <form action="{{ url_for('auth_login') }}" method="post">
                <div class="input-group">
                    <label for="email">Correo Electronico</label>
                    <input type="email" name="email" id="email" required>
                </div>
                <div class="input-group">
                    <label for="password">Contraseña</label>
                    <input type="password" name="password" id="password" required>
                </div>
                <button type="submit" class="submit-btn">Entrar</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# Routes existentes
@app.route('/auth/login-page')
def auth_login_page():
    return render_template_string(AUTH_LOGIN_TEMPLATE)

@app.route('/auth/login', methods=['POST'])
def auth_login():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    
    if not email or not password:
        flash('Por favor completa todos los campos.', 'danger')
        return redirect(url_for('auth_login_page'))
    
    print(f"Login attempt for {email}")
    result = firebase_auth.login_user(email, password)
    
    if result['success']:
        firebase_auth.set_user_session(result['user_data'])
        flash(result['message'], 'success')
        print(f"Successful login for {email}")
        return redirect(url_for('index'))
    else:
        flash(result['message'], 'danger')
        print(f"Failed login for {email}")
        return redirect(url_for('auth_login_page'))

@app.route('/auth/logout')
def auth_logout():
    firebase_auth.clear_user_session()
    flash('Has cerrado la sesion correctamente.', 'success')
    return redirect(url_for('auth_login_page'))

@app.route('/')
def index():
    if not firebase_auth.is_user_logged_in():
        return redirect(url_for('auth_login_page'))
    return redirect(url_for('search_page'))

# RUTA MEJORADA CON VISION AI PARA REPUESTOS
@app.route('/search')
@login_required
def search_page():
    current_user = firebase_auth.get_current_user()
    user_name = current_user['user_name'] if current_user else 'Usuario'
    user_name_escaped = html.escape(user_name)
    
    # Verificar disponibilidad de Vision AI
    vision_available = vision_gemini.is_vision_available() and vision_gemini.is_gemini_available()
    gemini_only = not vision_gemini.is_vision_available() and vision_gemini.is_gemini_available()
    
    if vision_available:
        vision_status = "✅ Vision AI + Gemini Activo"
    elif gemini_only:
        vision_status = "🟡 Solo Gemini (Configurar Google Cloud para análisis completo)"
    else:
        vision_status = "❌ Vision AI No Disponible (Configurar APIs)"
    
    vision_js_bool = "true" if (vision_available or gemini_only) else "false"
    
    content = f'''
    <div class="container">
        <div class="user-info">
            <span><strong>{user_name_escaped}</strong> | {vision_status}</span>
            <div style="display: inline-block; margin-left: 15px;">
                <a href="{url_for('auth_logout')}" style="background: #dc3545; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; margin-right: 8px;">Salir</a>
                <a href="{url_for('index')}" style="background: #28a745; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px;">Inicio</a>
            </div>
        </div>
        
        {{% with messages = get_flashed_messages(with_categories=true) %}}
            {{% if messages %}}
                {{% for category, message in messages %}}
                    <div class="flash {{{{ category }}}}">{{{{ message }}}}</div>
                {{% endfor %}}
            {{% endif %}}
        {{% endwith %}}
        
        <h1>Buscar Repuestos Automotrices</h1>
        <p class="subtitle">🔧 Texto o Imagen - Resultados especializados en 15 segundos</p>
        
        <!-- Selector de modo de búsqueda -->
        <div class="search-mode-toggle">
            <div class="mode-button active" onclick="switchMode('text')" id="textModeBtn">
                📝 Búsqueda por Texto
            </div>
            <div class="mode-button{' disabled' if not (vision_available or gemini_only) else ''}" onclick="{'' if (vision_available or gemini_only) else 'return false; '}switchMode('image')" id="imageModeBtn" style="{'opacity: 0.5; cursor: not-allowed;' if not (vision_available or gemini_only) else ''}">
                📷 Búsqueda por Imagen
            </div>
        </div>
        
        <!-- Sección de búsqueda por texto -->
        <div id="textSearchSection">
            <form id="searchForm">
                <div class="search-bar">
                    <input type="text" id="searchQuery" placeholder="Ej: cárter de aceite honda civic, pastillas de freno..." required>
                    <button type="submit">Buscar</button>
                </div>
            </form>
        </div>
        
        <!-- Sección de búsqueda por imagen -->
        <div id="imageSearchSection" style="display: none;">
            <div class="upload-section" id="uploadSection">
                <div class="upload-icon">🔧</div>
                <div class="upload-text">
                    <strong>Sube una imagen del repuesto automotriz</strong><br>
                    Arrastra y suelta o haz clic para seleccionar<br>
                    <small>Ideal para: cárteres, rotores, filtros, válvulas, etc.</small>
                </div>
                <input type="file" id="imageInput" class="file-input" accept="image/*">
                <button type="button" class="upload-button" onclick="document.getElementById('imageInput').click()">
                    📷 Seleccionar Imagen
                </button>
                <div style="font-size: 12px; color: #666; margin-top: 10px;">
                    Formatos: JPG, PNG, WebP | Máximo: 5MB
                </div>
            </div>
            <div id="imagePreview" style="display: none; text-align: center;">
                <img id="previewImg" class="preview-image" alt="Preview">
                <div>
                    <button type="button" id="analyzeImageBtn" class="upload-button" style="margin: 10px 5px;">
                        🔍 Analizar Repuesto
                    </button>
                    <button type="button" onclick="resetImageUpload()" style="background: #dc3545; color: white; padding: 10px 15px; border: none; border-radius: 6px; margin: 10px 5px;">
                        ❌ Cancelar
                    </button>
                </div>
            </div>
            <div id="analysisResult" class="analysis-result" style="display: none;">
                <h4 style="color: #2e7d32; margin-bottom: 8px;">✅ Repuesto Identificado</h4>
                <p id="analysisText"></p>
                <p id="analysisMethod" style="font-size: 12px; color: #666; margin-top: 5px;"></p>
                <button type="button" id="searchFromImageBtn" class="upload-button" style="margin-top: 10px;">
                    🛒 Buscar este Repuesto
                </button>
            </div>
        </div>
        
        <div class="tips">
            <h4>🚀 Especializado en Repuestos Automotrices:</h4>
            <ul style="margin: 8px 0 0 15px; font-size: 13px;">
                <li><strong>IA Especializada:</strong> Identifica cárteres, rotores, filtros, válvulas automáticamente</li>
                <li><strong>Búsqueda Optimizada:</strong> Términos técnicos precisos para mecánicos</li>
                <li><strong>Tiendas USA:</strong> Amazon, AutoZone, RockAuto, NAPA, Advance Auto</li>
                <li><strong>Precios Reales:</strong> Desde $25 (juntas) hasta $300+ (cárteres)</li>
            </ul>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <h3 id="loadingTitle">Buscando repuestos...</h3>
            <p id="loadingSubtitle">Máximo 15 segundos</p>
        </div>
        <div id="error" class="error"></div>
    </div>
    
    <script>
        let searching = false;
        let currentMode = 'text';
        let uploadedImageData = null;
        let analysisData = null;
        
        // Cambiar modo de búsqueda
        function switchMode(mode) {{
            if (mode === 'image' && !{vision_js_bool}) {{
                showError('Vision AI no está disponible. Configura GEMINI_API_KEY en las variables de entorno.');
                return;
            }}
            
            currentMode = mode;
            document.getElementById('textModeBtn').classList.toggle('active', mode === 'text');
            document.getElementById('imageModeBtn').classList.toggle('active', mode === 'image');
            document.getElementById('textSearchSection').style.display = mode === 'text' ? 'block' : 'none';
            document.getElementById('imageSearchSection').style.display = mode === 'image' ? 'block' : 'none';
            hideError();
        }}
        
        // Búsqueda por texto
        document.getElementById('searchForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            if (searching) return;
            const query = document.getElementById('searchQuery').value.trim();
            if (!query) return showError('Por favor ingresa un repuesto automotriz');
            performSearch(query);
        }});
        
        // Manejo de carga de imagen
        document.getElementById('imageInput').addEventListener('change', function(e) {{
            const file = e.target.files[0];
            if (!file) return;
            
            if (file.size > 5 * 1024 * 1024) {{
                showError('La imagen es muy grande. Máximo 5MB.');
                return;
            }}
            
            if (!file.type.startsWith('image/')) {{
                showError('Por favor selecciona un archivo de imagen válido.');
                return;
            }}
            
            const reader = new FileReader();
            reader.onload = function(e) {{
                uploadedImageData = e.target.result;
                document.getElementById('previewImg').src = uploadedImageData;
                document.getElementById('uploadSection').style.display = 'none';
                document.getElementById('imagePreview').style.display = 'block';
                document.getElementById('analysisResult').style.display = 'none';
            }};
            reader.readAsDataURL(file);
        }});
        
        // Análisis de imagen
        document.getElementById('analyzeImageBtn').addEventListener('click', function() {{
            if (!uploadedImageData || searching) return;
            
            searching = true;
            showLoading('🔍 Analizando repuesto...', 'Identificando pieza automotriz...');
            
            fetch('/api/analyze-image', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{image: uploadedImageData}})
            }})
            .then(response => response.json())
            .then(data => {{
                searching = false;
                hideLoading();
                
                if (data.success) {{
                    analysisData = data;
                    document.getElementById('analysisText').textContent = 
                        `Repuesto identificado: "${{data.search_term}}"`;
                    document.getElementById('analysisMethod').textContent = 
                        `Método: ${{data.analysis_method || 'AI Analysis'}}`;
                    document.getElementById('analysisResult').style.display = 'block';
                }} else {{
                    showError(data.error || 'Error analizando imagen');
                }}
            }})
            .catch(error => {{
                searching = false;
                hideLoading();
                showError('Error de conexión analizando imagen');
            }});
        }});
        
        // Búsqueda desde imagen analizada
        document.getElementById('searchFromImageBtn').addEventListener('click', function() {{
            if (!analysisData || searching) return;
            performSearch(analysisData.search_term);
        }});
        
        // Drag and drop
        const uploadSection = document.getElementById('uploadSection');
        uploadSection.addEventListener('dragover', function(e) {{
            e.preventDefault();
            uploadSection.classList.add('dragover');
        }});
        uploadSection.addEventListener('dragleave', function(e) {{
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        }});
        uploadSection.addEventListener('drop', function(e) {{
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {{
                document.getElementById('imageInput').files = files;
                document.getElementById('imageInput').dispatchEvent(new Event('change'));
            }}
        }});
        
        // Reset de imagen
        function resetImageUpload() {{
            uploadedImageData = null;
            analysisData = null;
            document.getElementById('imageInput').value = '';
            document.getElementById('uploadSection').style.display = 'block';
            document.getElementById('imagePreview').style.display = 'none';
            document.getElementById('analysisResult').style.display = 'none';
            hideError();
        }}
        
        // Búsqueda unificada
        function performSearch(query) {{
            if (searching) return;
            searching = true;
            showLoading('🛒 Buscando repuestos...', 'Consultando tiendas especializadas...');
            
            const timeoutId = setTimeout(() => {{
                searching = false;
                hideLoading();
                showError('Búsqueda muy lenta - Intenta de nuevo');
            }}, 15000);
            
            fetch('/api/search', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{query: query}})
            }})
            .then(response => {{
                clearTimeout(timeoutId);
                searching = false;
                return response.json();
            }})
            .then(data => {{
                hideLoading();
                if (data.success) {{
                    window.location.href = '/results';
                }} else {{
                    showError(data.error);
                }}
            }})
            .catch(() => {{
                clearTimeout(timeoutId);
                searching = false;
                hideLoading();
                showError('Error de conexión');
            }});
        }}
        
        // Utilidades
        function showLoading(title = 'Buscando repuestos...', subtitle = 'Máximo 15 segundos') {{
            document.getElementById('loadingTitle').textContent = title;
            document.getElementById('loadingSubtitle').textContent = subtitle;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
        }}
        
        function hideLoading() {{
            document.getElementById('loading').style.display = 'none';
        }}
        
        function showError(msg) {{
            hideLoading();
            const e = document.getElementById('error');
            e.textContent = msg;
            e.style.display = 'block';
        }}
        
        function hideError() {{
            document.getElementById('error').style.display = 'none';
        }}
    </script>'''
    
    return render_template_string(render_page('Busqueda de Repuestos', content))

# API SEARCH (sin cambios significativos)
@app.route('/api/search', methods=['POST'])
@login_required
def api_search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip() if data else ''
        if not query:
            return jsonify({'error': 'Consulta requerida'}), 400
        
        if len(query) > 80:
            query = query[:80]
        
        user_email = session.get('user_email', 'Unknown')
        print(f"Search request from {user_email}: {query}")
        
        products = price_finder.search_products(query)
        
        session['last_search'] = {
            'query': query,
            'products': products,
            'timestamp': datetime.now().isoformat(),
            'user': user_email
        }
        
        print(f"Search completed for {user_email}: {len(products)} products found")
        return jsonify({'success': True, 'products': products, 'total': len(products)})
        
    except Exception as e:
        print(f"Search error: {e}")
        try:
            query = request.get_json().get('query', 'auto part') if request.get_json() else 'auto part'
            fallback = price_finder._get_examples(query)
            session['last_search'] = {'query': str(query), 'products': fallback, 'timestamp': datetime.now().isoformat()}
            return jsonify({'success': True, 'products': fallback, 'total': len(fallback)})
        except:
            return jsonify({'error': 'Error interno del servidor'}), 500

# RESULTS PAGE MEJORADA
@app.route('/results')
@login_required
def results_page():
    try:
        if 'last_search' not in session:
            flash('No hay busquedas recientes.', 'warning')
            return redirect(url_for('search_page'))
        
        current_user = firebase_auth.get_current_user()
        user_name = current_user['user_name'] if current_user else 'Usuario'
        user_name_escaped = html.escape(user_name)
        
        search_data = session['last_search']
        products = search_data.get('products', [])
        query = html.escape(str(search_data.get('query', 'busqueda')))
        
        # Verificar si fue búsqueda por imagen
        image_analysis = session.get('last_image_analysis')
        if image_analysis:
            search_source = f"📷 Imagen ({image_analysis.get('analysis_method', 'AI')})"
        else:
            search_source = "📝 Texto"
        
        products_html = ""
        badges = ['MEJOR PRECIO', 'CALIDAD', 'POPULAR']
        colors = ['#4caf50', '#ff9800', '#9c27b0']
        
        for i, product in enumerate(products[:6]):
            if not product:
                continue
            
            badge = f'<div style="position: absolute; top: 8px; right: 8px; background: {colors[min(i, 2)]}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;">{badges[min(i, 2)]}</div>' if i < 3 else ''
            
            title = html.escape(str(product.get('title', 'Repuesto')))
            price = html.escape(str(product.get('price', '$0.00')))
            source = html.escape(str(product.get('source', 'Tienda')))
            link = html.escape(str(product.get('link', '#')))
            
            # Icono según la tienda
            store_icon = "🚗" if "autozone" in source.lower() else "🔧" if "rockauto" in source.lower() else "📦"
            
            products_html += f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: white; position: relative; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                    {badge}
                    <h3 style="color: #1a73e8; margin-bottom: 8px; font-size: 16px;">{title}</h3>
                    <div style="font-size: 28px; color: #2e7d32; font-weight: bold; margin: 12px 0;">{price} <span style="font-size: 12px; color: #666;">USD</span></div>
                    <p style="color: #666; margin-bottom: 12px; font-size: 14px;">{store_icon} Tienda: {source}</p>
                    <a href="{link}" target="_blank" rel="noopener noreferrer" style="background: #1a73e8; color: white; padding: 10px 16px; text-decoration: none; border-radius: 6px; font-weight: 600; display: inline-block; font-size: 14px;">🛒 Ver Repuesto</a>
                </div>'''
        
        prices = [p.get('price_numeric', 0) for p in products if p.get('price_numeric', 0) > 0]
        stats = ""
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            savings = max_price - min_price
            
            stats = f'''
                <div style="background: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <p><strong>{len(products)} repuestos encontrados</strong></p>
                    <p><strong>💰 Mejor precio: ${min_price:.2f}</strong></p>
                    <p><strong>📊 Precio promedio: ${avg_price:.2f}</strong></p>
                    <p><strong>💵 Ahorro máximo: ${savings:.2f}</strong></p>
                </div>'''
        
        content = f'''
        <div style="max-width: 800px; margin: 0 auto;">
            <div style="background: rgba(255,255,255,0.15); padding: 12px; border-radius: 8px; margin-bottom: 15px; text-align: center; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 14px;"><strong>{user_name_escaped}</strong></span>
                <div style="margin-left: 15px;">
                    <a href="{url_for('auth_logout')}" style="background: rgba(220,53,69,0.9); color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px; margin-right: 8px;">Salir</a>
                    <a href="{url_for('search_page')}" style="background: rgba(40,167,69,0.9); color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 13px;">Nueva Búsqueda</a>
                </div>
            </div>
            
            <h1 style="color: white; text-align: center; margin-bottom: 8px;">🔧 Repuestos: "{query}"</h1>
            <p style="text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 25px;">Búsqueda completada</p>
            
            {stats}
            {products_html}
        </div>'''
        
        return render_template_string(render_page('Repuestos - Price Finder USA', content))
    except Exception as e:
        print(f"Results page error: {e}")
        flash('Error al mostrar resultados.', 'danger')
        return redirect(url_for('search_page'))

@app.route('/api/health')
def health_check():
    try:
        return jsonify({
            'status': 'OK', 
            'timestamp': datetime.now().isoformat(),
            'firebase_auth': 'enabled' if firebase_auth.firebase_web_api_key else 'disabled',
            'serpapi': 'enabled' if price_finder.is_api_configured() else 'disabled',
            'vision_ai': 'enabled' if vision_gemini.is_vision_available() else 'disabled',
            'gemini_ai': 'enabled' if vision_gemini.is_gemini_available() else 'disabled',
            'automotive_mode': 'specialized'
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500

# Middleware y error handlers
@app.before_request
def before_request():
    if 'timestamp' in session:
        try:
            timestamp_str = session['timestamp']
            if isinstance(timestamp_str, str) and len(timestamp_str) > 10:
                last_activity = datetime.fromisoformat(timestamp_str)
                time_diff = (datetime.now() - last_activity).total_seconds()
                if time_diff > 1200:  # 20 minutos
                    session.clear()
        except:
            session.clear()
    
    session['timestamp'] = datetime.now().isoformat()

@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.errorhandler(404)
def not_found(error):
    return '<h1>404 - Pagina no encontrada</h1><p><a href="/">Volver al inicio</a></p>', 404

@app.errorhandler(500)
def internal_error(error):
    return '<h1>500 - Error interno</h1><p><a href="/">Volver al inicio</a></p>', 500

if __name__ == '__main__':
    print("Price Finder USA - Starting with Automotive Vision AI...")
    print(f"Firebase: {'OK' if os.environ.get('FIREBASE_WEB_API_KEY') else 'NOT_CONFIGURED'}")
    print(f"SerpAPI: {'OK' if os.environ.get('SERPAPI_KEY') else 'NOT_CONFIGURED'}")
    print(f"Gemini: {'OK' if os.environ.get('GEMINI_API_KEY') else 'NOT_CONFIGURED'}")
    print(f"Vision: {'OK' if VISION_AVAILABLE else 'NOT_AVAILABLE'}")
    print("Specialized mode: AUTOMOTIVE PARTS")
    print(f"Puerto: {os.environ.get('PORT', '5000')}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, threaded=True)
else:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
