import naive_nty


naive = naive_nty.NaiveNty(path='dataset_abante.json')
naive.train()
naive.test('abante_test', export=True)
