const methods = {
	init: async () => {
		if(typeof(cv) !== 'undefined') return { status: "loaded" }
		self.importScripts('./opencv.js')
		cv = await cv()
		return { status: "loaded" }
	},

	convertToGray: async (imageData) => {
		const img = cv.matFromImageData(imageData)
		
		let result = new cv.Mat()
		cv.cvtColor(img, result, cv.COLOR_BGR2GRAY)

		img.delete()
		return imageDataFromMat(result)
	}
}

//А эта функция переводит mat в imageData для canvas
function imageDataFromMat(mat) {
  // converts the mat type to cv.CV_8U
  const img = new cv.Mat()
  const depth = mat.type() % 8
  const scale =
    depth <= cv.CV_8S ? 1.0 : depth <= cv.CV_32S ? 1.0 / 256.0 : 255.0
  const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128.0 : 0.0
  mat.convertTo(img, cv.CV_8U, scale, shift)

  // converts the img type to cv.CV_8UC4
  switch (img.type()) {
    case cv.CV_8UC1:
      cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA)
      break
    case cv.CV_8UC3:
      cv.cvtColor(img, img, cv.COLOR_RGB2RGBA)
      break
    case cv.CV_8UC4:
      break
    default:
      throw new Error(
        'Bad number of channels (Source image must have 1, 3 or 4 channels)'
      )
  }
  const clampedArray = new ImageData(
    new Uint8ClampedArray(img.data),
    img.cols,
    img.rows
  )
  img.delete()
	mat.delete()
  return clampedArray
}

//В этой функции мы просто вызываем нужный метод и возвращаем ответ
onmessage = async function(e) {
	const { action, payload } = e.data

	const response = methods[action](payload)
	if(!response.then)
		postMessage({ action, payload: response })
	else
		response.then(response => {
			postMessage({ action, payload: response })
		})
}