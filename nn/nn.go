package nn

// golearn のデータ形式を使って gobrain で機械学習するパッケージ

import (
	"encoding/json"
	"errors"
	"os"

	"github.com/goml/gobrain"
	"github.com/sjwhitworth/golearn/base"
)

// ClassifierData : NN データの型
// [MEMO] resolveClassName と convData はデータ構造に依存する部分
type ClassifierData struct {
	ff               *gobrain.FeedForward                   //
	hiddenSize       int                                    //
	epochs           int                                    //
	lRate            float64                                // lRate --- learning rate
	mFactor          float64                                // mFactor --- momentum factor
	debug            bool                                   // debug --- debug mode for gobrain
	resolveClassName func([]float64) string                 // クラス名を決定する関数
	convData         func(base.FixedDataGrid) [][][]float64 // golearn のデータ形式から gobrain の形式に変換する関数
}

// NewClassifier : 分類器のインスタンスを作成
func NewClassifier(hiddenSize, epochs int, lRate, mFactor float64, debug bool, resolveClassName func([]float64) string, convData func(base.FixedDataGrid) [][][]float64) (r *ClassifierData) {
	nn := ClassifierData{
		hiddenSize:       hiddenSize,
		epochs:           epochs,
		lRate:            lRate,
		mFactor:          mFactor,
		debug:            debug,
		resolveClassName: resolveClassName,
		convData:         convData,
	}
	r = &nn
	return
}

// SetEpochs : epochs の設定
func (nn *ClassifierData) SetEpochs(v int) {
	nn.epochs = v
}

// SetLRate : lRate の設定
func (nn *ClassifierData) SetLRate(v float64) {
	nn.lRate = v
}

// SetMFactor : mFactor の設定
func (nn *ClassifierData) SetMFactor(v float64) {
	nn.mFactor = v
}

// SetResolveClassName : resolveClassName の設定
func (nn *ClassifierData) SetResolveClassName(f func([]float64) string) {
	nn.resolveClassName = f
}

// SetConvData : の設定
func (nn *ClassifierData) SetConvData(f func(base.FixedDataGrid) [][][]float64) {
	nn.convData = f
}

// Fit : 学習
func (nn *ClassifierData) Fit(bdiData base.FixedDataGrid) (err error) {

	if nn.convData == nil {
		err = errors.New("Fit: nn.convData is empty")
		return
	}

	trainData := nn.convData(bdiData)

	ff := &gobrain.FeedForward{}

	inSize := len(trainData[0][0])
	outSize := len(trainData[0][1])

	ff.Init(inSize, nn.hiddenSize, outSize)

	if nn.epochs == 0 {
		err = errors.New("Fit: nn.epochs is empty")
		return
	}
	if nn.lRate == 0 {
		err = errors.New("Fit: nn.lRate is empty")
		return
	}
	if nn.mFactor == 0 {
		err = errors.New("Fit: nn.mFactor is empty")
		return
	}
	ff.Train(trainData, nn.epochs, nn.lRate, nn.mFactor, nn.debug)

	nn.ff = ff

	return
}

// Predict : 予測
// [MEMO] 返り値となる予測値のデータ処理は golean/knn の Predict を参考にした。
// https://godoc.org/github.com/sjwhitworth/golearn/knn#KNNClassifier.Predict
func (nn *ClassifierData) Predict(data base.FixedDataGrid) (r base.FixedDataGrid, err error) {

	if nn.convData == nil {
		err = errors.New("Predict: nn.convData is empty")
		return
	}

	testData := nn.convData(data)
	ret := base.GeneratePredictionVector(data)

	for rowNo, d := range testData {
		x := d[0]            // 説明変数
		y := nn.ff.Update(x) // 目的変数

		// 目的変数に対応するクラス名を取得
		className := nn.resolveClassName(y)

		// rowNo 行目にクラス名を設定
		base.SetClass(ret, rowNo, className)
	}

	r = ret
	return
}

// Load : 分類器をファイルから読みだす
func Load(modelFile string) (r *ClassifierData, err error) {
	var f *os.File
	f, err = os.Open(modelFile)
	if err != nil {
		return
	}
	defer f.Close()

	ff := &gobrain.FeedForward{}
	err = json.NewDecoder(f).Decode(ff)
	if err != nil {
		return
	}

	r = &ClassifierData{
		ff: ff,
	}

	return
}

// Save : 分類器をファイルに保存する
func (nn *ClassifierData) Save(modelFile string) (err error) {
	var f *os.File
	f, err = os.Create(modelFile)
	if err != nil {
		return
	}
	defer f.Close()
	err = json.NewEncoder(f).Encode(nn.ff)
	return
}

/*
func calcScores(tp, fp, tn, fn int) (recall, precision, f1 float64) {
	precision = float64(tp) / float64(tp+fp)
	recall = float64(tp) / float64(tp+fn)
	f1 = 2.0 * precision * recall / (precision + recall)
	return
}
*/
