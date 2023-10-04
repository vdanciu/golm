package main

import (
	"encoding/json"
	"fmt"
	"vdanciu_lang_model/micrograd"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	mlp := micrograd.NewMLP(2, []int{9, 9, 1})
	// parse micrograd.MOON_X_JSON to a slice of slices of floats
	// (this is the dataset we'll train on)
	// parse micrograd.MOON_Y_JSON to a slice of ints
	err, x, y := getMoons()
	if err != nil {
		panic(err)
	}
	fmt.Println(mlp)
	fmt.Printf("number of parameters: %v\n", len(mlp.Parameters()))
	plotInputs(y, x)

	total_loss, accuracy := loss(mlp, x, y)
	fmt.Printf("initial loss: %v\n", total_loss)
	fmt.Printf("initial accuracy: %v\n", accuracy)

	// optimization
	for k := 0; k < 100; k++ {
		total_loss, accuracy := loss(mlp, x, y)

		// backward prop
		mlp.ZeroGrad()
		total_loss.Backward()

		// update (Stochastic Gradient Descent)
		learningRate := 1.0 - 0.9*float64(k)/100.0
		for _, p := range mlp.Parameters() {
			p.Data -= learningRate * p.Grad
		}

		fmt.Printf("step %v loss %v accuracy %v\n", k, total_loss, accuracy)
	}
}

func loss(mlp *micrograd.MLP, x [][]float64, y []int) (*micrograd.Value, float64) {
	// convert he inputs X to a slice of slices of micrograd.Values
	inputs := make([][]*micrograd.Value, len(x))
	for i := range inputs {
		x0 := micrograd.NewValue(x[i][0])
		x1 := micrograd.NewValue(x[i][1])
		inputs[i] = []*micrograd.Value{x0, x1}
	}

	// forward the inputs through the MLP
	scores := make([]*micrograd.Value, len(x))
	for i, input := range inputs {
		scores[i] = mlp.Forward(input)[0]
	}

	// compute the losses
	losses := make([]*micrograd.Value, len(x))
	for i := range losses {
		// losses[i] = ReLU(1 - scores[i]*y[i]) which is the max-margin loss used in SVMs
		losses[i] = micrograd.NewValue(1.0).Sub(scores[i].Mul(micrograd.NewValue(float64(y[i])))).Relu()
	}

	// compute the average loss
	avgLoss := micrograd.NewValue(0.0)
	for i := range losses {
		avgLoss = avgLoss.Add(losses[i])
	}
	avgLoss = avgLoss.Div(micrograd.NewValue(float64(len(losses))))

	// L2 regularization
	// this penalizes large model parameters by including them in the loss (squared but tempered by alpha)
	alpha := micrograd.NewValue(1e-4)
	reg_loss := micrograd.NewValue(0.0)
	for _, p := range mlp.Parameters() {
		reg_loss = reg_loss.Add(p.Mul(p))
	}
	reg_loss = reg_loss.Mul(alpha)
	// add the regularization loss to the average loss
	total_loss := avgLoss.Add(reg_loss)

	// compute accuracy
	correct := 0.0
	for i := range scores {
		if (scores[i].Data > 0) == (y[i] > 0) {
			correct += 1.0
		}
	}

	return total_loss, correct / float64(len(y))
}

func plotInputs(y []int, x [][]float64) {
	p := plot.New()

	p.Title.Text = "The moons"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	pts1 := make(plotter.XYs, 0)
	pts2 := make(plotter.XYs, 0)
	for i := range y {
		point := plotter.XY{X: x[i][0], Y: x[i][1]}
		if y[i] == -1 {
			pts1 = append(pts1, point)
		} else {
			pts2 = append(pts2, point)
		}
	}
	err := plotutil.AddScatters(p, "1", pts1, "2", pts2)
	if err != nil {
		panic(err)
	}

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "plot.png"); err != nil {
		panic(err)
	}
}

func getMoons() (error, [][]float64, []int) {
	var x [][]float64
	err := json.Unmarshal([]byte(micrograd.MOON_X_JSON), &x)
	if err != nil {
		panic(err)
	}

	var y []int
	err = json.Unmarshal([]byte(micrograd.MOON_Y_JSON), &y)
	if err != nil {
		panic(err)
	}
	return err, x, y
}
