const sourceImages = []
const angle = 30

const methods = {
	init: async () => {
		if(typeof(cv) !== 'undefined') return { status: "loaded" }
		self.importScripts('./opencv.js')

		cv = await cv()                                         //Таким образом мы инициализируем OpenCV

		orb = new cv.ORB(300)                                   //А затем сразу создадим ORB
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

		if(trainImage.keypoints.size() > 5)
			bfMatcher.match(queryImage.descriptors, trainImage.descriptors, matches)

		const good_matches = new cv.DMatchVector();
		for (let i = 0; i < matches.size(); i++) {
			if (matches.get(i).distance < 30) 
				good_matches.push_back(matches.get(i));
		}
		matches.delete()

		let finalImage = trainImage.image

		if(good_matches.size() > 5){
			const points1 = []
			const points2 = []

			for(let i = 0; i < good_matches.size(); i++) {
				points1.push(queryImage.keypoints.get(good_matches.get(i).queryIdx).pt.x)
				points1.push(queryImage.keypoints.get(good_matches.get(i).queryIdx).pt.y)
				points1.push(0)
		
				points2.push(trainImage.keypoints.get(good_matches.get(i).trainIdx).pt.x)
				points2.push(trainImage.keypoints.get(good_matches.get(i).trainIdx).pt.y)
			}

			const mat1 = cv.matFromArray(points1.length/3, 1, cv.CV_32FC3, points1);
			const mat2 = cv.matFromArray(points2.length/2, 1, cv.CV_32FC2, points2);

			const f = img.cols/2/(Math.tan(angle/2*Math.PI/180))
	
			const _mtx = [
				f, 0, img.cols / 2,
				0, f, img.rows / 2 ,
				0, 0, 1
			]

			const _dist = [ 0, 0, 0, 0 ]

			const mtx = cv.matFromArray(3, 3, cv.CV_64F, _mtx)
			const dist = cv.matFromArray(1, _dist.length, cv.CV_64F, _dist)

			const rvec = new cv.Mat()
			const tvec = new cv.Mat()

			cv.solvePnPRansac(mat1, mat2, mtx, dist, rvec, tvec)

			const rotationMatrix = new cv.Mat()
			cv.Rodrigues(rvec, rotationMatrix)

			const extrinsicMatrix = new cv.Mat(3, 4, cv.CV_64F)

			for(let i = 0; i < 3; i++){
				for(let j = 0; j < 3; j++){
					extrinsicMatrix.doublePtr(i, j)[0] = rotationMatrix.doubleAt(i, j)
				}
				extrinsicMatrix.doublePtr(i, 3)[0] = tvec.doubleAt(i, 0)
			}

			const zeros = cv.Mat.zeros(3, 3, cv.CV_64F)
			
			const projectionMatrix = new cv.Mat()
			cv.gemm(mtx, extrinsicMatrix, 1, zeros, 0, projectionMatrix)

			const _axis = [
				0, 0, 0, 1,
				30, 0, 0, 1,
				0, 30, 0, 1,
				0, 0, -30, 1
			]
			const axis = cv.matFromArray(4, 4, cv.CV_64F, _axis).t()

			const _points = new cv.Mat()
			cv.gemm(projectionMatrix, axis, 1, zeros, 0, _points)

			const points = _points.t()

			const pointsArr = []
			for(let i = 0; i < 4; i++){
				pointsArr.push({
					x: points.doubleAt(i, 0) / points.doubleAt(i, 2),
					y: points.doubleAt(i, 1) / points.doubleAt(i, 2)
				})
			}

	
			cv.line(finalImage, pointsArr[0], pointsArr[1], [ 255, 0, 0, 255 ], 2 )
			cv.line(finalImage, pointsArr[0], pointsArr[2], [ 0, 255, 0, 255 ], 2 )
			cv.line(finalImage, pointsArr[0], pointsArr[3], [ 0, 0, 255, 255 ], 2 )
			
			projectionMatrix.delete()
			axis.delete()
			points.delete()
			_points.delete()

			mat1.delete()
			mat2.delete()
			mtx.delete()
			dist.delete()

		}

		let result = new cv.Mat()
		cv.drawMatches(queryImage.image, queryImage.keypoints, finalImage, trainImage.keypoints, good_matches, result)

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