package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"vdanciu_lang_model/micrograd/migete"
)

func main() {
	RunNames()
}

func RunNames() {
	// read from the file "names.txt and create an array of strings
	words := readNames()
	fmt.Printf("%v\n", words[0:5])
	fmt.Printf("words in file: %v\n", len(words))

	// build the vocabulary of characters and mappings to and from integers
	vocab := buildVocabulary(words)
	fmt.Printf("%v\n", vocab)

	stoi := make(map[string]int)
	itos := make(map[int]string)
	for i, char := range vocab {
		stoi[char] = i
		itos[i] = char
	}
	fmt.Printf("%v\n", stoi)
	fmt.Printf("%v\n", itos)

	// create the dataset
	BLOCK_SIZE := 3
	fmt.Printf("block size: %v\n", BLOCK_SIZE)

	// seed the random number generator
	r := rand.New(rand.NewSource(42))
	// shuffle the words
	r.Shuffle(len(words), func(i, j int) { words[i], words[j] = words[j], words[i] })

	n1 := int(0.8 * float64(len(words)))
	n2 := int(0.9 * float64(len(words)))

	fmt.Printf("n1: %v\n", n1)
	fmt.Printf("n2: %v\n", n2)

	Xtr, Ytr := createDataset(words[0:n1], stoi, BLOCK_SIZE)
	Xval, Yval := createDataset(words[n1:n2], stoi, BLOCK_SIZE)
	Xte, Yte := createDataset(words[n2:], stoi, BLOCK_SIZE)

	fmt.Printf("Xtr: %v\n", Xtr.Shape())
	fmt.Printf("Ytr: %v\n", Ytr.Shape())
	fmt.Printf("Xval: %v\n", Xval.Shape())
	fmt.Printf("Yval: %v\n", Yval.Shape())
	fmt.Printf("Xte: %v\n", Xte.Shape())
	fmt.Printf("Yte: %v\n", Yte.Shape())

	VOCAB_SIZE := len(vocab)
	EMBEDDINGS_SIZE := 10
	HIDDEN_LAYER_SIZE := 200

	C := tensorRand[float32](VOCAB_SIZE, EMBEDDINGS_SIZE)
	W1 := tensorRand[float32](EMBEDDINGS_SIZE*BLOCK_SIZE, HIDDEN_LAYER_SIZE)
	b1 := tensorRand[float32](HIDDEN_LAYER_SIZE)
	W2 := tensorRand[float32](HIDDEN_LAYER_SIZE, VOCAB_SIZE)
	b2 := tensorRand[float32](VOCAB_SIZE)
	parameters := []*migete.Tensor[float32]{C, W1, b1, W2, b2}

	numParameters := 0
	for _, p := range parameters {
		numParameters += p.Size()
	}
	fmt.Printf("Number of parameters: %v\n", numParameters)

	// hyperparameters
	//LEARNING_RATE := 0.1
	NUM_EPOCHS := 100000
	//PRINT_EVERY := 1000

	// training loop
	for epoch := 0; epoch < NUM_EPOCHS; epoch++ {
		// minibatch
		MINIBATCH_SIZE := 32
		ix := migete.NewRandomUniformTensor[int](migete.MakeShape(MINIBATCH_SIZE), 0, Xtr.Shape()[0])
		inputs := Xtr.Index(ix)
		emb := C.Index(inputs)
		h := emb.Reshape(MINIBATCH_SIZE, EMBEDDINGS_SIZE*BLOCK_SIZE).Mul(W1).Add(b1).Tanh()
		logits := h.Mul(W2).Add(b2)
		loss := logits.CrossEntropy(Ytr.Index(ix))
	}
}

func createDataset(words []string, stoi map[string]int, blockSize int) (*migete.Tensor[int], *migete.Tensor[int]) {
	context := make([]int, blockSize)
	x := migete.NewEmptyTensor[int](migete.MakeShape(0, blockSize))
	y := migete.NewEmptyTensor[int](migete.MakeShape(0))
	for _, word := range words {
		word = word + "."
		for _, ch := range word {
			ix := stoi[string(ch)]
			x.Append(context)
			y.Append(migete.Flatten[int](ix))
			context = append(context[1:], ix)
		}
	}

	return x, y
}

func buildVocabulary(words []string) []string {
	mapVocab := make(map[string]struct{})
	for _, word := range words {
		for _, char := range word {
			if _, ok := mapVocab[string(char)]; !ok {
				mapVocab[string(char)] = struct{}{}
			}
		}
	}
	var vocab []string
	for char := range mapVocab {
		vocab = append(vocab, char)
	}
	vocab = append(vocab, ".")
	sort.Strings(vocab)
	return vocab
}

func readNames() []string {
	namesFile, err := os.Open("names.txt")
	if err != nil {
		panic(err)
	}
	defer namesFile.Close()

	var words []string
	scanner := bufio.NewScanner(namesFile)
	for scanner.Scan() {
		words = append(words, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
	return words
}

func tensorRand[T migete.Number](shape ...int) *migete.Tensor[T] {
	return migete.NewRandomTensor[T](shape)
}
