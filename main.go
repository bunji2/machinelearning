package main

import (
	"fmt"
	"os"

	"github.com/bunji2/machinelearning/irisdata"
	"github.com/bunji2/machinelearning/nn"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

const (
	irisDataFile = "iris.csv" // IRISデータのパス
	dataHeader   = false      // ヘッダの有無 (true - あり, false - なし)
)

func main() {
	os.Exit(run())
}

func run() int {

	// IRISデータの取得
	rawData, err := base.ParseCSVToInstances(irisDataFile, dataHeader)

	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	//fmt.Println(rawData)

	// データセットを学習用とテスト様に分離
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	err = evalNN(trainData, testData)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}

	err = evalKNN(trainData, testData)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 2
	}

	return 0
}

// evalNN : NN による分析
func evalNN(trainData, testData base.FixedDataGrid) (err error) {
	// 分類器の初期化
	// hiddenSize, epochs, lRate, mFactor, debug, resolveClassName, convData
	cls := nn.NewClassifier(100, 10000, 0.4, 0.6, false, irisdata.ResolveClassName, irisdata.ConvData)

	// 学習
	err = cls.Fit(trainData)
	if err != nil {
		return
	}

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
