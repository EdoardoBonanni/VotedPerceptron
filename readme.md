# Voted-perceptron

Implementazione dell’algoritmo voted perceptron descritto in Freund & Schapire 1999 e riproduzione di risultati analoghi a quelli riportati nella sezione 5 dell’articolo (in particolare
i grafici per d = 1 e d = 2 della figura 2) ma utilizzando il dataset Zalando al posto di MNIST.

## Link utili

[**Large Margin Classification Using the Perceptron Algorithm**](https://link.springer.com/content/pdf/10.1023/A:1007662407062.pdf) <br />
[**Zalando research**](https://github.com/zalandoresearch/fashion-mnist)

### Dataset

| Name  | Content | Examples | Size | Link |
| --- | --- |--- | --- |--- |
| `train-images-idx3-ubyte.gz`  | training set images  | 60,000| 26 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)|
| `train-labels-idx1-ubyte.gz`  | training set labels  | 60,000| 29 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)|
| `t10k-images-idx3-ubyte.gz`  | test set images  | 10,000| 4.3 MBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)|
| `t10k-labels-idx1-ubyte.gz`  | test set labels  | 10,000| 5.1 KBytes | [Download](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)|

## Prerequisiti

Ambiente di sviluppo che supporta Python3.
Richiesta l'installazione nell'ambiente di sviluppo dei package:

	- Numpy: package fondamentale per lo scentific computing con Python.		
	
	- matplotlib: libreria per la creazione di grafici per il linguaggio di programmazione Python e la libreria matematica NumPy.
		
## Code

utils -> zalandoReader.py (alcune parti delle funzioni sono riprese da [Zalando research](https://github.com/zalandoresearch/fashion-mnist)):	

	- funzione load_zalando_train: legge il dataset per l'apprendimento a partire da un path ricevuto in ingresso dalla funzione.

	- funzione load_zalando_test: legge il dataset per i test a partire da un path ricevuto in ingresso dalla funzione.

utils -> helper.py (alcune parti delle funzioni sono riprese da [Zalando research](https://github.com/zalandoresearch/fashion-mnist)):
	
	- funzione create_sprite_image: serve per creare l'immagine del dataset.

	- funzione vector_to_matrix_mnist: serve per creare la matrice contenente gli elementi del dataset.
	
	- funzione invert_grayscale: inverte la scala di grigi dell'immagine.
	
	- funzione get_sprite_image: serve per visualizzare dell'immagine del dataset.
	
configs.py

	- contiene i path per leggere i dataset.
	
usefulFunctions.py

	- funzione createArray:	crea un vettore contenente un numero di array vuoti pari al valore intero passato come parametro allal funzione.
	
	- funzione polynomialExpansion: permette di svolgere l'espansione polinomiale.
	
	- funzione predictLabelWithPolExp: predice la label attraverso l'espansione polinomiale.
	
	- funzione predLabelPolExprl: predice la label attraverso l'espansione polinomiale fino al passo rl (utilizzata per la predizione random).
	
	- funzione predictLabelStandard: predice la label nel modo standard.
	
	- funzione predLabelStandardrl: predice la label nel modo standard fino al passo rl (utilizzata per la predizione random)
	
VotedPerceptron.py

	- classe VotedPerceptronC: contiene le informazioni e la funzione per svolgere il training del dataset utilizzando l'algoritmo Voted-perceptron
		init: riceve come parametri il dataset (images e labels), il numero di epoche, il kernel degree, il valore da cui iniziare il ciclo di training e il passo del ciclo infine istanzia i vettori di predizione (v) e il vettore (c) che conta il numero di predizioni corrette.
		training: esegue il training sul dataset e restituisce i vettori di predizione e i vettori c calcolati ad ogni epoca.
		predict: calcola la label predetta a partire dall'istanza x.

training.py

	- funzione trainingClasses01: permette di eseguire il training su tutte le epoche e le classi del dataset ridotto (solo del kernel degree passato come parametro alla funzione).
	
	- funzione trainingClasses: permette di eseguire il training su tutte le epoche e le classi del dataset completo (solo del kernel degree passato come parametro alla funzione).
	
	- funzione trainingD: esegue tutti i training di uno specifico kernel degree passato come parametro alla funzione.
	
predictions.py

	- funzione lastUnnormalized: permette di fare la predizione della label attraverso il metodo last unnormalized.
	
	- funzione vote: permette di fare la predizione della label attraverso il metodo vote.
	
	- funzione avgUnnormalized: permette di fare la predizione della label attraverso il metodo average unnormalized.
	
	- funzione randomUnnormalized: permette di fare la predizione della label attraverso il metodo random unnormalized.
	
	- funzione lastNormalized: permette di fare la predizione della label attraverso il metodo last normalized.
	
	- funzione avgNormalized: permette di fare la predizione della label attraverso il metodo average normalized.
	
	- funzione randomNormalized: permette di fare la predizione della label attraverso il metodo random normalized.
	
	- funzione methodPredictionsGeneral: permette di fare la predizione su tutti i metodi unnormalized, normalized e vote, restituisce un array contenente i risultati di ogni predizione.
	
	- funzione methodPredictions: permette di fare la predizione su tutti i metodi unnormalized e vote, restituisce un array contenente i risultati di ogni predizione.
	
test.py

	- funzione prepareTest01: permette di eseguire i test su tutte le epoche e le classi del dataset ridotto (solo del kernel degree passato come parametro alla funzione). Restituisce il vettore contenente gli errori di predizione in percentuale fatti attraverso i vari metodi di predizione del file predictions.py (metodi unnormalized e vote)                     	
	
	- funzione prepareTest: permette di eseguire i test su tutte le epoche e le classi del dataset completo (solo del kernel degree passato come parametro alla funzione). Restituisce il vettore contenente gli errori di predizione in percentuale fatti attraverso i vari metodi di predizione del file predictions.py (metodi unnormalized e vote)       
							
	- funzione prepareTestComplete01: permette di eseguire i test su tutte le epoche e le classi del dataset ridotto (solo del kernel degree passato come parametro alla funzione). Restituisce il vettore contenente gli errori di predizione in percentuale fatti attraverso i vari metodi di predizione del file predictions.py                   	
	
	- funzione prepareTestComplete: permette di eseguire i test su tutte le epoche e le classi del dataset completo (solo del kernel degree passato come parametro alla funzione). Restituisce il vettore contenente gli errori di predizione in percentuale fatti attraverso i vari metodi di predizione del file predictions.py
									
	- funzione allTest: esegue tutti i test di uno specifico kernel degree passato come parametro alla funzione e chiama la funzione plotGraphic per creare il grafico.
	
	- funzione plotGraphic: crea il grafico.
	
main.py

	- funzione main: Legge il dataset di apprendimento e di testing, crea le immagini contenenti del dataset di apprendimento e di testing, infine esegue il training, il test e stampa il grafico per il kernel degree 1 e 2.
	
	
	
	
	
	
	


