from flask import Flask, request
import naive_nty
import json
import os
from os import walk
import matplotlib.pyplot as plt

# endpoint
class ApiNty:
    def __init__(self,naive_path = '' ,run_endpoints = False):
        self.app = Flask(__name__)

        self.run_endpoints = run_endpoints
        self.stack = []
        
        if naive_path:
            naive = naive_nty.NaiveNty()
            naive.import_model(naive_path)
            self.naive = naive
            if run_endpoints:
                self.endpoints()
                self.run()

     # --------------------------->

    def endpoints(self):
        @self.app.route('/classify', methods = ['POST'])
        def classify():
            item = request.json
            try:
                text = item['textBody']
            except KeyError:
                text = None
            try:
                subtitle = item['subtitle']
            except KeyError:
                subtitle = None
            try:
                title = item['title']
            except KeyError:
                title = None
                
            #we remove url encoded spaces of url by replacing '%20'by ''
            url = item['url'].replace('%20','')
            urlDict = {'Energía': ['energia', 'energy'], 'animales': ['animal', 'animales', 'naturaleza', 'ciencia', 'nature', 'science'], 'espacio': ['espacio', 'space', 'galaxia', 'astronomia', 'astronomy'], 'clima': ['clima', 'tiempo', 'weather'], 'famosos': ['famosos', 'famous', 'prensarosa', 'corazon'], 'futbol': ['futbol', 'football', 'soccer', 'premierleague', 'champions', 'laliga', 'europaleague'], 'tenis': ['tenis', 'wimbledon', 'tennis', 'rolandgarros', 'australianopen', 'lavercup', 'usopen'], 'baloncesto': ['baloncesto', 'basket', 'basketball', 'nba', 'eurobasket', 'ligaendesa', 'euroliga', 'eurocup', 'fiba'], 'formula 1': ['formula1', 'pilotos', 'gp', 'fia'], 'motociclismo': ['motociclismo', 'motogp', 'moto2', 'moto3'], 'Sector Inmobiliario': ['sectorinmobiliario', 'inmobiliaria', 'viviendas', 'pisos', 'hipoteca', 'alquiler'], 'StartUp': ['startup', 'empresaemergente', 'entrepeneur', 'emprendedor', 'emprendimiento'], 'Empresa': ['empresa', 'company', 'compañia', 'sl', 'negocio', 'business', ''], 'criptomonedas': ['criptomonedas', 'crypto', 'bitcoin', 'kucoin', 'binance', 'blockchain', 'altcoin', 'coinbase', 'smartcontract', 'token', 'wallet'], 'mercados': ['economia', 'finanzas', 'dinero', 'mercados', 'euro', 'dolar', 'libra', 'divisas', 'bce', 'fmi', 'fed', 'ibex', 'nasdaq', 'sp500', 'inversion', 'stocks', 'nikei', 'money', 'market', 'shares', 'value', ''], 'sucesos': ['sucesos', 'hechos', 'acontecimientos'], 'viajes': ['viajes', 'travel', 'trip', 'vacaciones', 'viajar', 'turismo'], 'cosmetica': ['cosmetica', 'cosmetico', 'cosmetics', 'maquillaje', 'embellecedor', 'crema', 'tratamiento', 'tratamientofacial', 'tratamientocutáneo', 'retoque'], 'nutricion': ['nutricion', 'nutrition', 'alimentacion', 'dieta', 'dietetica', 'proteinas', 'azucares', 'nutrientes', 'procesado', 'ultraprocesado', 'realfood'], 'Música': ['musica', 'music', 'spotify', 'rap', 'trap', 'concierto', 'festival', 'pop', 'dancehall', 'productor', 'cantante', 'instrumentos', 'rock', 'heavymetal'], 'cine y series': ['cine', 'series', 'netflix', 'hbo', 'filmin', 'disney', 'prime', 'actrices', 'actores', 'movistar', 'produccion', 'warner', 'universal', 'got', 'narcos', 'calamar', 'entretenimiento', 'movies', 'comedia', 'accion', 'drama', 'biografia', 'monografia', 'documental', 'hechos', 'historia', 'actualidad', 'hulu', 'hallmark', 'pelis', 'cinesa', 'cinefilo', 'backstage', 'palomitas', 'chill', ''], 'libros': ['Cientificos', 'Literatura', 'lingüistica', 'Biografias', 'Monografias', 'Recreativos', 'Poesia', 'Juveniles', 'Ficcion', 'Comedia', 'negocio', 'libros', 'business', 'autoayuda', 'historia', 'management', 'producto', 'liderazgo', 'superacion', 'ciencia', 'terror', 'negra', 'negro', 'policiaco', 'audiolibro', 'relatos', 'reverte', 'king', 'zafon', 'allende', 'benavent', 'mola', 'planeta', 'nobel', 'premio', 'psicologia', 'sicologia', 'follet', 'animacion', 'ux', 'ui', 'libros', 'books'], 'moviles': ['pc', 'tablet', 'hacking', 'informatica', 'programacion', 'javascript', 'tecnologia', 'python', 'ia', 'iot', 'tech', 'gadgets', 'componentes', 'computing', 'computer', 'hackers', 'moviles', 'mobile', ''], 'informatica': ['pc', 'tablet', 'hacking', 'informatica', 'programacion', 'javascript', 'tecnologia', 'python', 'ia', 'iot', 'tech', 'gadgets', 'componentes', 'computing', 'computer', 'hackers', 'moviles', 'mobile', ''], 'opinión': ['opinion', 'columnas', 'firmas', 'escritores', 'columnistas', 'articulistas', 'articulos', 'bustos', 'trueba', 'jabois', 'lindo', 'sostres', 'pozo', 'leila', 'lucas', 'simon', ''], 'sociedad': ['sociedad', 'personas', 'sucesos', 'noticias', 'general', 'espana', ''], 'Generalistas': ['actualidad', 'sucesos', 'noticias', 'generalistas', 'resumen', 'dia', 'news'], 'Deportes': ['futbol', 'baloncesto', 'basket', 'deportes', 'badminton', 'tenis', 'natacion', 'padel', 'surf', 'formula1', 'motogp', 'motos', 'coches', 'pesca', 'caza', 'toros', 'esports', 'gaming', 'fifa', 'champions', 'premier', 'laliga', 'Almería', 'AthleticClub\nAtleticodeMadrid', 'Barcelona', 'CeltadeVigo', 'Cadiz\nElche', 'Espanyol', 'Getafe', 'Girona', 'Mallorca', 'Osasuna\nRayoVallecano', 'RealBetis', 'RealMadrid', 'RealSociedad', 'RealValladolid', 'Sevilla', 'ValenciaCF', 'Vllarreal', 'nba', 'wimbledon', 'lakers', ''], 'Economía': ['economia', 'finanzas', 'dinero', 'mercados', 'euro', 'dolar', 'libra', 'divisas', 'bce', 'fmi', 'fed', 'ibex', 'nasdaq', 'sp500', 'inversion', 'stocks', 'nikei', 'money', 'market', 'shares', 'value', ''], 'Salud': ['salud', 'bienestar', 'autoayuda', 'estilo_de_vida', 'meditación', 'yoga', 'mindfulness', 'sanidad', 'health', 'medicina', 'farmacos', 'farmacia', 'farmaceutico', ''], 'Ocio y Cultura': ['libros', 'teatro', 'escritores', 'estrenos', 'festivales', 'ocio', 'cultura', 'actores', 'cine', 'autores', 'actrices', 'guionista', 'planes', 'entretenimiento', ''], 'Corazón': ['famosos', 'cotilleo', 'corazon', 'celebrities', 'famosas', 'salvame', 'telecinco', 'hola', 'pronto', 'quore', 'lecturas', 'posados', 'pantoja', 'matamoros', 'prensa_rosa'], 'Ciencia y Naturaleza': ['viajes', 'ciencia_naturaleza', 'naturaleza', 'ciencia', 'science', 'nature', 'bosques', 'biosfera', 'ecosistema', 'co2', 'oxigeno', 'cambio_climatico', 'tierra', 'planeta', 'ecologia', 'ecologismo', 'sostenibilidad', 'medioambiente', 'life', 'vida', 'volcanes', 'oceanos', 'mares', 'montañas', ''], 'Internacional': ['internacional', 'eeuu', 'francia', 'alemania', 'italia', 'china', 'rusia', 'america', 'europa', 'asia', 'democratas', 'republicanos', 'international', 'usa', ''], 'Prensa Local': ['local', 'prensa_local', 'pueblos', 'municipios', 'comarcas', 'provincias', ''], 'Motor': ['motos', 'coches', 'motogp', 'formula1', 'mercedes', 'ferrari', 'porsche', 'mclaren', 'maserati', 'cilindrada', 'cilindros', 'mecanica', 'dakar', 'rallies', 'rally', 'automovil'], 'Tecnología': ['portatiles', 'mac', 'iphone', 'android', 'google', 'tecnologia', 'bigdata', 'machinelearning', 'data', 'algoritmos', 'gadgets', 'werables', 'tech', 'technology', 'apple', 'microsoft', '5g', 'iot', 'ux', 'ui', 'diseno'], 'Diseño': ['Interiorismo', 'arte', 'decoración', 'diseno', 'arquitectura', 'casas', 'design', 'art', 'minimalismo', 'barroco', ''], 'Moda': ['complementos', 'bolsos', 'zapatos', 'moda', 'estilismo', 'maquillaje', 'fashion', 'dress', 'diseñadores', 'pasarela', 'joyas', 'looks'], 'LifeStyle': ['lifestyle', 'lujo', 'viajes', 'cosmética', 'wellness', 'barcos', 'yates', 'luxury', 'estilo', 'naturaleza', 'joyas', 'escapadas'], 'Gastronomia': ['gastronomia', 'food', 'foodie', 'vegano', 'flexiteriano', 'nutricion', 'keto', 'plantbased', 'cocteles', 'vegetarianismo', 'dieta', 'alimentacion'], 'politica': ['politica', 'gobierno', 'congreso', 'politics', 'senado', 'tribunalsupremo', 'constitucional', 'fiscalia', 'pp', 'psoe']}
            category_by_url = naive_nty.categorizeByUrl(url,urlDict)


            c_text = ''
            
            
            if text:
                c_text = text
            elif title:
                c_text = title
            elif subtitle:
                c_text = subtitle
             
            
            if len(c_text) < 10:
                json_response = json.dumps({'naive':None,'url':category_by_url, 'inter': category_by_url})
                return json_response    
            

            # inter = ''
            category_probs,safe = self.naive.categorize(c_text,safe_categorization=True) 
            # category_probs_top = {k: category_probs[k] for k in list(category_probs.keys())[:10]}
            # print('top 2 categories ---> ',category_probs_top)
            # print('text hint ---->',text[:150]) if text else print('subtitle hint ----->',subtitle[:150]) if subtitle else print('title hint ----->',title[:150])

            inter = [cat for cat in list(category_probs) if cat in category_by_url]
            json_response = json.dumps({'naive':category_probs, 'safe': safe, 'url':category_by_url,'inter':inter})
            return json_response

    def run(self):
        from waitress import serve
        print('running endpoints...')
        serve(self.app,host='0.0.0.0', port=1420)
