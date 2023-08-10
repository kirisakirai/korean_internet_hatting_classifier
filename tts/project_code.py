def model_load():
    import pickle
    import os

    path = os.getcwd()
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    vect_name = os.path.join(path, 'vect.pickle')
    model_name = os.path.join(path, 'model.pickle')

    with open(model_name, 'rb') as handle:
        model = pickle.load(handle)
    with open(vect_name, 'rb') as handle:
        vect_morp = pickle.load(handle)

    return model, vect_morp

def project_test_code(sentence, model, vect_morp):
    import rhinoMorph
    rn = rhinoMorph.startRhino()

    train_morphed = []
    train_morphed.append(rhinoMorph.onlyMorph_list(rn, sentence, pos=['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True))
    train_X_join_model = [" ".join(sentence) for sentence in train_morphed]

    vect_sentence = vect_morp.transform(train_X_join_model)
    vect_sentence_astype = vect_sentence.astype('float32')

    pred = model.predict(vect_sentence_astype)
    return pred