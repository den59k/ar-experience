const sourceImages = []

const methods = {
	init: async () => {
		if(typeof(cv) !== 'undefined') return { status: "loaded" }
		self.importScripts('./opencv.js')

		cv = await cv()                                         //Таким образом мы инициализируем OpenCV

		orb = new cv.ORB(500)                                   //А затем сразу создадим ORB
		bfMatcher = new cv.BFMatcher(cv.NORM_HAMMING2, true)    //А заодно и матчер

		console.log(Object.keys(cv))

		return { status: "loaded" }
	},

	convertToGray: async (imageData) => {
		const img = cv.matFromImageData(imageData)
		
		let result = new cv.Mat()
		cv.cvtColor(img, result, cv.COLOR_BGR2GRAY)

		img.delete()
		return imageDataFromMat(result)
	},

	loadSourceImage: async (imageData) => {
		const img = cv.matFromImageData(imageData)
		const keypointsData = getImageKeypoints(img)
		console.log(keypointsData)
		sourceImages.push(keypointsData)

		return { id: sourceImages.length-1 }
	},

	matchPoints: async ({ sourceImage, imageData }) => {
		const img = cv.matFromImageData(imageData)

		const queryImage = sourceImages[sourceImage.id]
		const trainImage = getImageKeypoints(img)

		const matches = new cv.DMatchVector()
		bfMatcher.match(queryImage.descriptors, trainImage.descriptors, matches)

		const good_matches = new cv.DMatchVector();
		for (let i = 0; i < matches.size(); i++) {
			if (matches.get(i).distance < 30) 
				good_matches.push_back(matches.get(i));
		}
		matches.delete()

		let result = new cv.Mat()
		cv.drawMatches(queryImage.image, queryImage.keypoints, trainImage.image, trainImage.keypoints, good_matches, result)

		trainImage.delete()
		good_matches.delete()

		return imageDataFromMat(result)
	},

	getCameraPosition: async (imageData) => {
		
	}
}

function getImageKeypoints(image){
	const imgGray = new cv.Mat()
	cv.cvtColor(image, imgGray, cv.COLOR_BGR2GRAY)

	const keypoints = new cv.KeyPointVector()             //Ключевые точки
	const none = new cv.Mat() 
	const descriptors = new cv.Mat()                              //Дескрипторы точек (т.е. некое уникальное значение)

	orb.detectAndCompute(imgGray, none, keypoints, descriptors)

	none.delete()
	imgGray.delete()

	const dispose = () => {
		keypoints.delete()
		descriptors.delete()
		image.delete()
	}

	return { image, keypoints, descriptors, delete: dispose }
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