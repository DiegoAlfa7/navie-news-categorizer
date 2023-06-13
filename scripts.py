
import naive_nty
import json
from datetime import datetime
from os import walk
import api
import shutil

naive = naive_nty.NaiveNty()
naive.import_model('./0.8692_word_prior.json')

naive.plot_freqs(type='json')

# api_ = api.ApiNty()
# api_.run()

#stack test ------>
# keywords = {'CIENCIA_NATURALEZA': ['animales','naturaleza','ciencia','rayo','science','nature','bosques','biosfera','ecosistema','co2','oxigeno','climatico','tierra','planeta','ecologia','ecologismo','sostenibilidad','medioambiente','life','vida','volcanes','oceanos','mares','montañas'],
# 'CULTURA': ['libros','teatro','escritores','estrenos','festivales','ocio','cultura','actores','cine','autores','actrices','guionista','planes','entretenimiento'],
# 'DEPORTES': ['futbol','baloncesto','basket','deportes','badminton','tenis','natacion','padel','surf','formula','motogp','motos','coches','pesca','caza','toros','esports','gaming','fifa','champions','premier','liga','Almería','Athletic','Atletico','Barcelona','Celta','Cadiz','Elche','Espanyol','Getafe','Girona','Mallorca','Osasuna','Vallecano','Real','Betis','Madrid','Sociedad','Valladolid','Sevilla','Valencia','Vllarreal','nba','wimbledon','lakers'],
# 'DISENO': ['interiorismo','arte','decoración','diseno','arquitectura','casas','design','art','minimalismo','barroco','románico','diseño'],
# 'ECONOMIA': ['economia','finanzas','dinero','mercados','euro','dolar','libra','divisas','bce','fmi','fed','ibex','nasdaq','sp500','inversion','stocks','nikei','money','market','shares','valor','puntos'],
# 'FAMOSOS': ['famosos','famous','Falcó','Carbonero','Pantoja','Esteban','Borbones','Borbón','Shakira','Bertin','Onieva','Hilton','Urdangarin','Ordoñez','Matamoros','Igartiburu','Supervivientes','Salvame','Bosé','Kardashian','Spears','Bieber','Gaga','Rihanna','Drake','Freeman','Pacino','Bunny','Anuel','Balvin','Rauw','Maluma','Niro','Arguiñano','Bezos','Musk','Gates','Buffet','Zuckerberg','Pichai','Rogan','Dorsey','Rahm','Pausini','Mcgregor','Mayweather','Tyson','Presley','Boyer','Goirigolzarri','Echevarria','Bustamante','Bisbal','Rosalia','Aitana','Patiño','Deluxe','Pombo','Escanes','Mejide','Markle','Hadid','Minaj','Jam','Ratajkowski','Padilla','Yatra','Jenner','Brosnan','Beyonce','Perry','Gallagher','Lipa','DeGeneres','Schwarzenegger','Stallone','Gibson','Hamm','Kanye','Sarda','Torroja','Vaquerizo','Chenoa','Peluso','Onega','Galvez','Cooper','Shayk','Levine','Barrymore','Camilla','Hardy','Brady','Letizia','Agag','Eastwood','Pataky','Norris'],
# 'GASTRONOMIA': ['gastronomia','food','foodie','vegano','flexiteriano','nutricion','keto','plantbased','cocteles','vegetarianismo','dieta','alimentacion','comida','cena','desayuno','especias'],
# 'GENERALISTAS': ['actualidad','sucesos','noticias','generalistas','resumen','dia','news'],
# 'INTERNACIONAL': ['Trump','Biden','Putin','Merkel','Macron','Pen','Berlusconi','Sassoli','Leyen','Michel','Borrell','Clinton','Kiev','Moscú','China','Pekín','Tokyo','Japón','Europea','Europa','Ucrania','Zelenski','Irak','América','ONU','EEUU','Taliban','Afganistán','Venezuela','Kirchner','Renzi','Johnson','Erdogan','Maduro','Chavez','Scholz','Zaporiyia','Muqtada','Morawiecki','Cisjordania','Bolsonaro','Inglaterra',' UK','Delhi','Vaticano','Rotterdam','Orban','Tripoli','Hungría','Kosovo','Tebboune','Mariupol','Donetsk','Riga','Lenin','Uribe','Obrador','AMLO','MPLA','Obama','Bush','Seúl','Jong-un','Jinping','Pelosi','internacional','Francia','Alemania','Italia','China','Rusia','América','Europa','Asia','África','Oceanía','demócratas','republicanos','internacional','USA'],
# 'LIFESTYLE': ['lifestyle','lujo','viajes','cosmética','wellness','barcos','yates','luxury','estilo','naturaleza','joyas','escapadas'],
# 'MODA': ['complementos','bolsos','zapatos','moda','estilismo','maquillaje','fashion','dress','diseñadores','pasarela','joyas','looks'],
# 'MOTOR': ['moto','coche','motogp','formula','formula1','Mercedes','Ferrari','Porsche','Mclaren','Maserati','cilindrada','cilindros','mecanica','Dakar','rallies','rally','automovil','carrocería','caballos','chasis','volante','frenos','llantas'],
# 'POLITICA': ['politica','gobierno','congreso','politics','senado','tribunal','supremo','constitucional','fiscalia','PP','PSOE','Podemos','VOX','Ciudadanos','ERC','PACMA','diputado','alcalde','fiscal','senador','presidente','edil','Audiencia'],
# 'PRENSA_LOCAL': ['local','pueblos','municipios','comarcas','provincias','Álava','Albacete','Alacant','Almería','Ávila','Badajoz','Illes Balears','Barcelona','Burgos','Cáceres','Cádiz','Castelló','Córdoba','Coruña','Cuenca','Girona','Granada','Guadalajara','Gipuzkoa','Huelva','Huesca','Jaén','León','Lleida','Rioja','Lugo','Madrid','Málaga','Murcia','Nafarroa','Ourense','Asturias','Palencia','Palmas','Pontevedra','Salamanca','Tenerife','Cantabria','Segovia','Sevilla','Soria','Tarragona','Teruel','Toledo','Valéncia','Valladolid','Bizkaia','Zamora','Zaragoza','Ceuta','Melilla'],
# 'SALUD': ['salud','bienestar','autoayuda','vida','meditación','yoga','mindfulness','sanidad','health','medicina','farmacos','farmacia','farmaceutico','vacuna','tratamiento','remedio','contagio','prevención','síntoma'],
# 'SUPERHEROES': ['marvel','dc','superheroes','comic','batman','superman','ironman','antman','Deadpool','Hulk','Wolverine','spiderman','superpoderes','avengers','vengadores','stark','aquaman','thanos','star','wars','disney','darth','vader','skywalker'],
# 'TECNOLOGIA': ['portatiles','mac','iphone','android','google','tecnologia','data','algoritmos','gadgets','werables','tech','technology','apple','microsoft','5g','iot','ux','ui','diseno','headset','ordenadores'],
# 'ANIMALES': ['animal','animales','naturaleza','ciencia','nature','science','aves','extinción','mamífero','mascota','carnívoro','hervíboro','marsupial','safari','hembra','macho','especie','mascota','hábitat'],
# 'BALONCESTO': ['baloncesto','basket','basketball','nba','eurobasket','liga','endesa','euroliga','eurocup','fiba','Thompson','Leonard','Williamson','Murray','Simmons','Irving','Booker','Butler','George','Doncic','Embiid','LeBron','James','Jordan','Jokic','Curry','Durant','Antetokounmpo','Lakers','Celtics','Raptors','Thunder','Pistons','Spurs','Heat','Timberwolves','Mavericks','Knicks','Bulls','Pelicans','Grizzlies','Cavaliers','76ers','Nets','Bulls','Warriors','ACB','NBA','canasta'],
# 'CINE_SERIES': ['cine','series','netflix','hbo','filmin','disney','prime','actrices','actores','movistar','produccion','warner','universal','got','narcos','calamar','entretenimiento','movies','comedia','accion','drama','biografia','monografia','documental','hechos','historia','actualidad','hulu','hallmark','pelis','cinesa','cinefilo','backstage','palomitas','chill'],
# 'CRIPTOMONEDAS': ['criptomonedas','crypto','bitcoin','kucoin','binance','blockchain','altcoin','coinbase','smart','contract','token','wallet','ethereum','merge','cardano','kraken','nft','staking','dogecoin','cardano','ADA','ETH','BTC','USDT','BNB','BUSD','SOL','DOGE','MATIC','SHIB'],
# 'EMPRESA': ['empresa','company','compañia','sl','negocio','business','franquicia'],
# 'ENERGIA': ['energia','gas','energy','hidrogeno','electricidad','nuclear','oil','petroleo','renovable','eolico','eolica','biomasa','solar','fosil','carbon','electrico','panel','geotermica','placa','calentador','combustible','fotovoltaica'],
# 'FORMULA_1': ['formula','pilotos','fia','Alonso','Hamilton','Sainz','Verstappen','Leclerc','Vettel','Ferrari','Alpine','Mclaren'],
# 'FUTBOL': ['futbol','futbolista','liverpool','football','soccer','champions','liga','Messi','Ronaldo','Florentino','Benzema','Hazard','Courtuois','Barça','Atleti','Betis','Alavés','Depor','CD','FC','UD','Real','Madrid','Mbape','Guardiola','gol','League','Zidane','Ancelotti','Simeone','Emery','Lewandoski','Umtiti','Varane','Valdano','Isco','Vinicius','Oblak','Fekir','Modric','Neymar','FIFA','UEFA','Haaland','Salah','Pogba','Depay','Iniesta','Dybala','Juventus','Bayern','Borussia','Lukaku','Koke'],
# 'LIBROS': ['cientificos','literatura','lingüistica','biografias','monografias','recreativos','poesia','juveniles','ficcion','comedia','negocio','libros','business','autoayuda','historia','management','producto','liderazgo','superacion','ciencia','terror','negra','negro','policiaco','audiolibro','relatos','reverte','king','zafon','allende','benavent','mola','planeta','nobel','premio','psicologia','sociología','follet','animacion','ux','ui','libros','books'],
# 'MERCADOS': ['economia','finanzas','dinero','mercados','euro','dolar','libra','divisas','bce','fmi','fed','ibex','nasdaq','inversion','stocks','nikei','money','market','shares','value'],
# 'MOTOCICLISMO': ['motociclismo','motogp','moto','trial','motrocross','Agostini','Rossi','Nieto','Marquez','Miller','Nakagami','Pedrossa'],
# 'MUSICA': ['musica','music','spotify','rap','trap','concierto','festival','pop','dancehall','productor','cantante','instrumentos','rock','gira','disco','Canserbero','Sabaton','Disturbed','Avicii','Guetta','Bunny','Skrillex','Morodo','Eminem','Rihanna','Sitana','Sabina','Bieber'],
# 'S_INMOBILIARIO': ['inmobiliario','inmobiliaria','viviendas','pisos','hipoteca','alquiler','inmueble','hacienda','propiedad'],
# 'SOCIEDAD': ['sociedad','personas','sucesos','noticias','general','espana'],
# 'STARTUP': ['startup','empresa','emergente','entrepeneur','emprendedor','emprendimiento','ronda','financiación','seed','incubadora'],
# 'SUCESOS': ['viajes','travel','trip','vacaciones','viajar','turismo'],
# 'TENIS': ['tenis','wimbledon','tennis','roland garros','cup','open','Nadal','Djokovic','Federer','Alcaraz','Medvedev','Zverev','Tsitsipas','Ruud','Berrettini','Kyrgios','Tiafoe','Opelka','Monfils','Goffin','Swiatek','Kontaveit','Sakkari','Badosa','Jabeur','Sabalenka','Halep','Kasatkina','Kudermetova','Kvitova','atp','masters','challenger','sock','slam','tenista','Norrie'],
# 'VIAJES': ['viajes','travel','trip','vacaciones','viajar','turismo']}

# lemmasDict =  {}

# lemmasNaive = naive_nty.NaiveNty(posTypes=['NOUN', 'PROPN'])

# for c in keywords:
#     lemmasDict[c] = {}
#     lemmasDict[c]['processed_text'] = lemmasNaive.process_text(' '.join(keywords[c]))
#     for t in keywords[c]:
#         # print("'"+t+"'")
#         token = lemmasNaive.nlp(t)[0]
#         lemmasDict[c][t] = {'text': token.text, 'lemma': token.lemma_, 'pos': token.pos_}

# serialized = json.dumps(lemmasDict, indent=4)

# with open('./assets/data/keywords_lemmas.json', mode='w', encoding='UTF-8') as file_:
#     file_.write(serialized)
#     file_.close()



# naive = naive_nty.NaiveNty('category','textBody','new_naive_dataset_limit_40.json',posTypes=['NOUN', 'PROPN'])
# naive.train()
# naive.test(test_name='NOUN_PROPN_with_keys_06_word_prior',keywords = keywords, keywords_bias=0.6, export=True)


# naive = naive_nty.NaiveNty('category','textBody','new_naive_dataset_limit_40.json',posTypes=['NOUN', 'PROPN'])
# naive.train()
# naive.test(test_name='NOUN_PROPN_without_keys', export=True)


# # ----------------------------------------------


# naive = naive_nty.NaiveNty('category','textBody','new_naive_dataset_limit_40.json',posTypes=['NOUN', 'PROPN', 'VERB', 'ADJ'])
# naive.train()
# naive.test(test_name='NOUN_PROPN_VERB_ADJ_with_keys_06',keywords = keywords, keywords_bias=0.6, export=True)


# naive = naive_nty.NaiveNty('category','textBody','new_naive_dataset_limit_40.json',posTypes=['NOUN', 'PROPN', 'VERB', 'ADJ'])
# naive.train()
# naive.test(test_name='NOUN_PROPN_VERB_ADJ_without_keys', export=True)



# naive = naive_nty.NaiveNty(class_field='category',text_field='textBody',posTypes=['NOUN', 'PROPN'],spacy_model='es_core_news_lg')
# naive.import_model('./NOUN_PROPN_with_keys_06/0.8615384615384616_2022-09-27T19:44/model/2022-09-27T19:44:50.878933_0.8615384615384616.json')
# print(naive.categorize('futbol, la mayor goleada desde que bajo yisus el real madrid gana la champions'))

# # compute accuracy
# file = open('./naive_dataset_27092022.json', encoding='UTF-8')
# list_all = json.load(file)

# y = [a.get('category').lower().strip() for a in list_all]
# acc_test = naive.compute_accuracy(3,'textBody',list_all,y)
# print('accuracy test all ---->',acc_test)

# naive.export('./NOUN_PROPN_with_keys_06/')




# api_ = api.ApiNty()

# api_ = api.ApiNty('./NOUN_PROPN_no_keys/0.8538461538461538_2022-09-27T13:52/model/2022-09-27T13:52:03.903722_0.8538461538461538.json',run_endpoints=True)
# api_ = api.ApiNty('./NOUN_PROPN_with_keys_06/0.8615384615384616_2022-09-27T19:44/model/2022-09-27T19:44:50.878933_0.8615384615384616.json',run_endpoints=True)



# generate plot ------------------>

# api_.plot_category_words_freq_from_model()
# shutil.rmtree('./naive_categories_plot_freq/', ignore_errors=False, onerror=None)

# ------------------>

# single train ------------------>

# api_.train_naive()

# ------------------>

# stack train ------------------>

# api_.train_naive_stack('./stack',200,3,0.75)
# print('total freqs :',api_.compute_average_prediction('el madrid es la cima del futbol, tiene sentido poner fin aqui a esta experienciael italiano carlo ancelotti técnico del real madrid, aseguró que se retirará una vez termine su etapa, decisión que cree tiene sentido ya que el real madrid es la cima del futbol'))

# ------------------>





# ------------------------------------------------------------------------------------------------------



# api_ = api.ApiNty(auto_run = False)

# api_.train_naive_stack('./stack20082022/',250,2,0.8)

# api_.plot_category_words_freq_from_model()

# api_.import_naive_stack('./stack20082022/')
# api_.endpoints()
# api_.run()
# print('total freqs :',api_.compute_average_prediction('el madrid es la cima del futbol, tiene sentido poner fin aqui a esta experienciael italiano carlo ancelotti técnico del real madrid, aseguró que se retirará una vez termine su etapa, decisión que cree tiene sentido ya que el real madrid es la cima del futbol'))


# naive = naive_nty.NaiveNty()
# naive.lemmatize()
# naive.train()
# naive.compute_accuracy(2)
# naive.export('./exports/')
# naive.import_model('./exports/1660484342.891685_0.7027027027027027.json')
# print(naive.categorize('futbol, la mayor goleada desde que bajo yisus el real madrid gana la champions'))

# arr = []

# for i in range(100):
#     naive = naive_nty.NaiveNty()
#     naive.lemmatize([5,15])
#     naive.train()
#     report = naive.compute_accuracy(2)
#     accuracy, categories, predicted = report['accuracy'],report['categories'],report['predicted']

#     item = {"accuracy": accuracy,"categories":categories,"predicted":predicted, "train_data": naive.train_data }
#     #"mean": sum(list(map(lambda x: x['accuracy'], arr)))/(i+1)
#     # arr.append(item)
#     json_string = json.dumps(item, indent=2)
#     print(json_string)
#     with open(str('./json/'+str(datetime.now().timestamp())+'_'+str(i)+'_'+str(accuracy)+'.json'), mode='w', encoding='UTF-8') as file_:
#         file_.write(json_string)
#         file_.close()

# def getMeanAccuracyFromFiles(path = './json'):
#     filenames =  [file for file in walk(path)][0][2]
#     accuracies = [float((acc.split('_')[-1:][0])[:len((acc.split('_')[-1:][0])) -5])for acc in filenames]
#     # print('sum ------->', sum(accuracies))
#     # print('len ------->', len(accuracies))
#     mean = sum(accuracies)/len(accuracies)
#     return mean

# path = './exports'
# print('Directory :', path, '\nMean accuracy from files: ',getMeanAccuracyFromFiles(path))



