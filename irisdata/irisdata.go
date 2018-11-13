package irisdata

// gobrain 専用。
// IRISデータに依存する部分を吸収するパッケージ
// このファイルを修正することで他の形式のデータに対応

import "github.com/sjwhitworth/golearn/base"

// classNames : IRISのクラス名のリスト
var classNames = []string{
	"Iris-setosa",     // cid = 0
	"Iris-versicolor", // cid = 1
	"Iris-virginica",  // cid = 2
}

// ResolveClassName : クラス名を解決する関数
// convFromClassNameToFloat64Array と対をなす
func ResolveClassName(y []float64) string {
	cid := convFromFloat64ArrayToClassID(y)
	return classNames[cid]
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
		// [MEMO] 次は i 行目のクラス名を取り出しているのだが、慣れないと非常にわかりにくい。
		//        Get() と GetStringFromSysVal() の仕様が難解でクソ過ぎる。
		//        ----
		//        多分、次に見返した時には覚えていないと思うのでもう少しメモしておこう。
		//        Get : i 行目の、指定した AttributeSpec の値を []byte 形式で取り出す。
		//        GetStringFromSysVal : Attribute 依存の []byte 形式の値を文字列で取り出す。
		//        ----
		//        "attrSpecs[cols-1]","attrs[cols-1]" は最後尾の列にクラス名がくるので
		//        最後尾の列の AttributeSpec/Attribute を参照している。
		//        ----
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
	case "Iris-setosa":
		return []float64{1, 0, 0, 0}
	case "Iris-versicolor":
		return []float64{0, 1, 0, 0}
	case "Iris-virginica":
		return []float64{0, 0, 1, 0}
	}
	// Other
	return []float64{0, 0, 0, 1}
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
