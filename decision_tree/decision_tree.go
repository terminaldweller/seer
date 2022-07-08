package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Class int32

const (
	classOne Class = 0
	classTwo       = 1
)

type pointType struct {
	x     float32
	y     float32
	class Class
}

func generateRandData(n int32, points []pointType) {
	rand.Seed(time.Now().UnixNano())
	var i int32
	for i = 0; i < n; i++ {
		dirX := rand.Float32()
		dirY := rand.Float32()
		if dirX > 0.5 {

			points[i].x = rand.Float32() * 20
		} else {
			points[i].x = -rand.Float32() * 20
		}
		if dirY > 0.5 {
			points[i].y = rand.Float32() * 20
		} else {
			points[i].y = -rand.Float32() * 20
		}
		class := rand.Float32()
		if class < 0.5 {
			points[i].class = classOne
		} else {
			points[i].class = classTwo
		}
	}
}

func decisionTreeV1(points []pointType) {
	for i := 0; i < len(points); i++ {
		if points[i].x <= -12 {
			if points[i].x <= 9 {
				if points[i].y < 9 {

				} else {

				}
			} else {

			}
		} else {

		}
	}
}

func calcClassProbs(points []pointType, classCount int) []float64 {
	var counts = make([]int, classCount)
	for i := 0; i < len(points); i++ {
		counts[points[i].class]++
	}

	var probs = make([]float64, classCount)
	for i := 0; i < classCount; i++ {
		probs[i] = float64(counts[i]) / float64(len(points))
	}

	return probs
}

func calcEntropy(probs []float64) float64 {
	var entropy float64

	for i := 0; i < len(probs); i++ {
		entropy += probs[i] * float64(math.Log2(probs[i]))
	}

	entropy = entropy * -1
	return entropy
}

func main() {
	var n int32 = 1000
	var points = make([]pointType, n)
	generateRandData(n, points)
	var i int32
	for i = 0; i < n; i++ {
		fmt.Println(points[i].x, ":", points[i].y, ":", points[i].class)
	}
	probs := calcClassProbs(points, 2)
	for i := 0; i < 2; i++ {
		fmt.Println(probs[i])
	}
	entropy := calcEntropy(probs)
	fmt.Println(entropy)
}
