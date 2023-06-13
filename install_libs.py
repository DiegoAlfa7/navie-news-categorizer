import subprocess

models = [
    # 'en_core_web_lg',
    'es_core_news_lg'
    ]
libs = [
    'json',
    'unidecode',
    'flask',
    'waitress',
    'spacy',
    're',
    'random',
    'trax',
    'matplotlib',
    'pymongo',
    'zyte-autoextract',
    'scrapy',
    'scrapy-autoextract',
    'gspread',
    'sentry_sdk',
    'dataclasses',
    'dataclasses_json']

def install_libraries(verbose=True):

    print('Importing necessary libs...')
    lib = [['pip', 'install', a] for a in libs]
    model = [['python3','-m','spacy','download',spacy_model] for spacy_model in models]
    # libs_import = [['pip','install','json'],['pip','install','unidecode'],['pip','install','flask'],['pip','install','waitress'],['pip','install','spacy'],['pip','install','re'],['pip','install','random'],['pip','install','trax'],['pip','install','matplotlib'],['python3','-m','spacy','download',spacy_model]]
    libs_import = lib + model
    for idx, i in enumerate(libs_import):
        log = subprocess.run(i, capture_output=True)
        print(' '.join(libs_import[idx]), ' --->  OK!')

    # libs_import_log = [subprocess.run(i, capture_output=True) for i in libs_import]
    # print('Libs imported! |||||| log ------->  ', libs_import_log)
    print('Libs imported! |||||| log ------->  ', log) if verbose else None

install_libraries()
