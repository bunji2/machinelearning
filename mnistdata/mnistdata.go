package mnistdata

import (
	"fmt"

	mnist "github.com/petar/GoMNIST"
	"github.com/sjwhitworth/golearn/base"
)

// classNames : MNISTのクラス名のリスト
var classNames = []string{
	"0", // cid = 0
	"1", // cid = 1
	"2", // cid = 2
	"3", // cid = 3
	"4", // cid = 4
	"5", // cid = 5
	"6", // cid = 6
	"7", // cid = 7
	"8", // cid = 8
	"9", // cid = 9
}

// Load : MNIST データの読み出し
func Load(imageFile, labelFile string, max int) (r base.FixedDataGrid, err error) {
	var data *mnist.Set
	data, err = mnist.ReadSet(imageFile, labelFile)
	if err != nil {
		return
	}
	//fmt.Println("NRow =", data.NRow)
	//fmt.Println("NCol =", data.NCol)
	//fmt.Println("len(Images) =", len(data.Images))
	//fmt.Println("len(Labels) =", len(data.Labels))
	rowSize := len(data.Images)
	if max < rowSize {
		rowSize = max
	}
	colSize := data.NRow*data.NCol + 1
	n := base.NewDenseInstances()
	attrSpecs := make([]base.AttributeSpec, colSize)
	for j := 0; j < colSize-1; j++ {
		name := fmt.Sprintf("%d", j)
		attrSpecs[j] = n.AddAttribute(base.NewFloatAttribute(name))
	}
	//classAttribute := base.NewFloatAttribute(fmt.Sprintf("%d", colSize-1))
	//attrSpecs[colSize-1] = n.AddAttribute(classAttribute)

	classAttribute := base.NewCategoricalAttribute()
	classAttribute.SetName(fmt.Sprintf("%d", colSize-1))
	attrSpecs[colSize-1] = n.AddAttribute(classAttribute)
	err = n.AddClassAttribute(classAttribute)
	if err != nil {
		return
	}

	err = n.Extend(rowSize)
	if err != nil {
		return
	}

	for i := 0; i < rowSize; i++ {
		//fmt.Print(i, ":")
		for j, b := range data.Images[i] {
			n.Set(attrSpecs[j], i, base.PackFloatToBytes(float64(b)))
			//fmt.Print(b, ",")
		}
		//n.Set(attrSpecs[colSize-1], i, base.PackFloatToBytes(float64(data.Labels[i])))
		n.Set(attrSpecs[colSize-1], i, classAttribute.GetSysValFromString(fmt.Sprintf("%d", data.Labels[i])))
	}
	r = n
	//fmt.Println(n)
	return
}

// ResolveClassName : クラス名を解決する関数
// convFromClassNameToFloat64Array と対をなす
func ResolveClassName(y []float64) string {
	cid := convFromFloat64ArrayToClassID(y)
	if cid >= 0 && cid < len(classNames) {
		return classNames[cid]
	}
	return "unknown"
}

// ConvData : golearn の FixedDataGrid 形式のデータを gobrain の形式に変換する関数。IRIS 専用
//            gobrain のデータは [][][]float64 となる。
//            [..., [x, y], ...] の形で、説明変数 x 目的変数 y ともに []float64 となる。
func ConvData(rawData base.FixedDataGrid) (r [][][]float64) {

	// [MEMO] cols は列の数、rows は行の数。列の数は後で出てくる Attribute の数でもある。
	cols, rows := rawData.Size()

	// [MEMO] あとで FixedDataGrid 内の各値を取り出すのに各列の AttributeSpec が必要となるので調べておく。
	attrs := rawData.AllAttributes()
	attrSpecs := make([]base.AttributeSpec, cols)
	for j := 0; j < cols; j++ {
		attrSpecs[j], _ = rawData.GetAttribute(attrs[j])
	}

	r = make([][][]float64, rows)
	for i := 0; i < rows; i++ {
		// x : 説明変数
		x := []float64{}
		for j := 0; j < cols-1; j++ { // ← cols-1 番目の列はクラス名なので、その手前まで。
			// [MEMO] FixedDataGridでは各値は []byte になっており、float64 の値は以下のようにして取り出す。
			//        base.UnpackBytesToFloat : []byte 形式の浮動小数点の値を float64 で取り出す。
			v := base.UnpackBytesToFloat(rawData.Get(attrSpecs[j], i))
			x = append(x, v)
		}
		//-----------------------------------------------------

		className := attrs[cols-1].GetStringFromSysVal(rawData.Get(attrSpecs[cols-1], i))

		// y : 目的変数。クラス名を []float64 に変換
		y := convFromClassNameToFloat64Array(className)

		// 説明変数 x と目的変数 y を i 行に設定
		r[i] = [][]float64{x, y}
	}
	return
}

// convFromClassNameToFloat64Array : クラス名を[]float64に変換する関数
func convFromClassNameToFloat64Array(className string) []float64 {
	switch className {
	case "0":
		return []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	case "1":
		return []float64{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	case "2":
		return []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	case "3":
		return []float64{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}
	case "4":
		return []float64{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
	case "5":
		return []float64{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}
	case "6":
		return []float64{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}
	case "7":
		return []float64{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}
	case "8":
		return []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}
	case "9":
		return []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}
	}
	// Other
	return []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
}

// convFromFloat64ArrayToClassID : []float64をクラスIDに変換する関数
func convFromFloat64ArrayToClassID(y []float64) (cid int) {
	cid = 0
	max := y[0]
	for i := 1; i < len(y); i++ {
		if y[i] > max {
			max = y[i]
			cid = i
		}
	}
	return
}
