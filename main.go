package main

import (
	"fmt"
	"os"
	"time"

	"github.com/bunji2/machinelearning/irisdata"
	"github.com/bunji2/machinelearning/mnistdata"
	"github.com/bunji2/machinelearning/nn"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

const (
	irisDataFile   = "dataset/iris.csv"                   // IRISデータのパス
	dataHeader     = false                                // ヘッダの有無 (true - あり, false - なし)
	mnistImageFile = "dataset/train-images-idx3-ubyte.gz" //
	mnistLabelFile = "dataset/train-labels-idx1-ubyte.gz" //
)

func main() {
	os.Exit(run())
}

func run() int {

	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s [knn | nn] [ iris | mnist ]", os.Args[0])
		return 1
	}

	cmd := os.Args[1]
	data := os.Args[2]

	var err error
	if cmd == "knn" && data == "iris" {
		err = knnIRIS()
	} else if cmd == "knn" && data == "mnist" {
		err = knnMNIST()
	} else if cmd == "nn" && data == "iris" {
		err = nnIRIS()
	} else if cmd == "nn" && data == "mnist" {
		err = nnMNIST()
	} else {
		err = fmt.Errorf("unknown args cmd = %s data = %s", cmd, data)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}

	return 0
}

// knnIRIS : KNN を用いた評価
func knnIRIS() (err error) {
	var rawData base.FixedDataGrid

	// IRISデータの取得
	rawData, err = base.ParseCSVToInstances(irisDataFile, dataHeader)

	if err != nil {
		return
	}

	// データセットを学習用とテスト様に分離
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	err = evalKNN(trainData, testData)
	return
}

// nnIRIS : NN を用いた評価
func nnIRIS() (err error) {
	var rawData base.FixedDataGrid

	// IRISデータの取得
	rawData, err = base.ParseCSVToInstances(irisDataFile, dataHeader)

	if err != nil {
		return
	}

	// データセットを学習用とテスト様に分離
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	err = evalNN(trainData, testData, irisdata.ResolveClassName, irisdata.ConvData)
	return
}

func knnMNIST() (err error) {
	var rawData base.FixedDataGrid
	t1 := time.Now()
	rawData, err = mnistdata.Load(mnistImageFile, mnistLabelFile, 10000)
	t2 := time.Now()
	fmt.Println("elapse time of loading :", t2.Sub(t1))
	if err != nil {
		return
	}

	// データセットを学習用とテスト用に分離
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.1)
	t3 := time.Now()
	fmt.Println("elapse time of Spliting :", t3.Sub(t2))

	cols, rows := trainData.Size()
	fmt.Printf("trainData = %d x %d\n", rows, cols)
	cols, rows = testData.Size()
	fmt.Printf("testData = %d x %d\n", rows, cols)

	err = evalKNN(trainData, testData)
	return
}

func nnMNIST() (err error) {
	var rawData base.FixedDataGrid
	t1 := time.Now()
	rawData, err = mnistdata.Load(mnistImageFile, mnistLabelFile, 1000)
	t2 := time.Now()
	fmt.Println("elapse time of loading :", t2.Sub(t1))
	if err != nil {
		return
	}

	// データセットを学習用とテスト用に分離
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.1)
	t3 := time.Now()
	fmt.Println("elapse time of Spliting :", t3.Sub(t2))

	cols, rows := trainData.Size()
	fmt.Printf("trainData = %d x %d\n", rows, cols)
	cols, rows = testData.Size()
	fmt.Printf("testData = %d x %d\n", rows, cols)

	err = evalNN(trainData, testData, mnistdata.ResolveClassName, mnistdata.ConvData)
	return
}

// evalNN : NN による分析
func evalNN(trainData, testData base.FixedDataGrid, resolveClassName func([]float64) string, convData func(base.FixedDataGrid) [][][]float64) (err error) {
	// 分類器の初期化
	// hiddenSize, epochs, lRate, mFactor, debug, resolveClassName, convData
	//cls := nn.NewClassifier(100, 10000, 0.4, 0.6, false, irisdata.ResolveClassName, irisdata.ConvData)
	cls := nn.NewClassifier(100, 5000, 0.4, 0.6, true, resolveClassName, convData)

	// 学習
	t1 := time.Now()
	err = cls.Fit(trainData)
	if err != nil {
		return
	}
	t2 := time.Now()
	fmt.Println("elapsed time to fit :", t2.Sub(t1))

	// 予測
	var predictions base.FixedDataGrid
	predictions, err = cls.Predict(testData)
	if err != nil {
		return
	}
	//fmt.Println(predictions)
	t3 := time.Now()
	fmt.Println("elapsed time to predict :", t3.Sub(t2))

	// スコア算出
	var confusionMat map[string]map[string]int
	confusionMat, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		return
	}

	fmt.Println("NN (gobrain)")
	fmt.Println(evaluation.GetSummary(confusionMat))

	// 保存
	cls.Save("model.json")
	return
}

// evalKNN : kNN による分析
func evalKNN(trainData, testData base.FixedDataGrid) (err error) {
	// KNNを初期化
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	// 学習
	cls.Fit(trainData)

	// 予測
	var predictions base.FixedDataGrid
	predictions, err = cls.Predict(testData)
	if err != nil {
		return
	}

	//fmt.Println(predictions)

	// スコア算出
	var confusionMat map[string]map[string]int
	confusionMat, err = evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		return
	}

	fmt.Println("kNN (golearn)")
	fmt.Println(evaluation.GetSummary(confusionMat))
	return
}
