metrics = []

def save_metrics( model_id, metrics_bundle ):
    if model_id < len(metrics):
        metrics[model_id] = metrics_bundle
    else:
        metrics.append(metrics_bundle)
    return

def test( model_id, dataset_id ):
    global models, datasets, metrics

    model = models[model_id]
    df = datasets[dataset_id]

    X_test = df.iloc[:,1:21].values
    y = df.iloc[:,0].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    y_test = pd.get_dummies(y1).values

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    save_metrics(model_id, {'accuracy' : accuracy, 'actual': actual.tolist(), 'predicted':predicted.tolist()})

    return( metrics[model_id] )
