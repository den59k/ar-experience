const sourceImages = []
const lastFramesData = []
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

		sourceImages.push(keypointsData)

		return { id: sourceImages.length-1 }
	},

	estimateCameraPosition: async ({ id, imageData }) => {
		const img = cv.matFromImageData(imageData)
		
		const queryImage = sourceImages[id]
		const trainImage = getImageKeypoints(img)
		
		let finalImage = new cv.Mat()
		cv.cvtColor(trainImage.image, finalImage, cv.COLOR_GRAY2RGB)

		trainImage.delete()

		return imageDataFromMat(finalImage)
	},

	calculatePnP: async ({ sourceImage, imageData }) => {
		const img = cv.matFromImageData(imageData)
		const queryImage = sourceImages[sourceImage.id]
		const trainImage = getImageKeypoints(img)

		let queryPoints;
		let trainPoints;

		let finalImage = new cv.Mat()
		cv.cvtColor(trainImage.image, finalImage, cv.COLOR_GRAY2RGB)

		if(lastFramesData[sourceImage.id]){
			queryPoints = lastFramesData[sourceImage.id].queryPoints
			trainPoints = lastFramesData[sourceImage.id].trainPoints

			const lastPoints = cv.matFromArray(trainPoints.length, 1, cv.CV_32FC2, trainPoints.flat());
			const nextPoints = new cv.Mat()

			const status = new cv.Mat()
			const errors = new cv.Mat()

			cv.calcOpticalFlowPyrLK(lastFramesData[sourceImage.id].image, trainImage.image, lastPoints, nextPoints, status, errors)
	
			trainPoints = []
			for(let i = 0; i < nextPoints.rows; i++)
				if(status.charAt(i) === 1 && errors.floatAt(i) < 10)
					trainPoints.push([
						nextPoints.floatAt(i, 0),
						nextPoints.floatAt(i, 1)
					])
			
			errors.delete()
			status.delete()
			nextPoints.delete()

			if(trainPoints.length < 6){
				trainPoints = null
				queryPoints = null
				lastFramesData[sourceImage.id] = null
			}
		}

		if(!trainPoints){
			const matches = new cv.DMatchVector()

			if(trainImage.keypoints.size() > 5)
				bfMatcher.match(queryImage.descriptors, trainImage.descriptors, matches)

			const good_matches = []
			for (let i = 0; i < matches.size(); i++) {
				if (matches.get(i).distance < 30) 
					good_matches.push(matches.get(i));
			}
			matches.delete()

			if(good_matches.length > 8){
				queryPoints = []
				trainPoints = []
				
				for(let i = 0; i < good_matches.length; i++) {
					queryPoints.push([
						queryImage.keypoints.get(good_matches[i].queryIdx).pt.x,
						queryImage.keypoints.get(good_matches[i].queryIdx).pt.y,
						0
					])

					trainPoints.push([
						trainImage.keypoints.get(good_matches[i].trainIdx).pt.x,
						trainImage.keypoints.get(good_matches[i].trainIdx).pt.y
					])
				}
			}
		}

		if(trainPoints){
			const mat1 = cv.matFromArray(queryPoints.length, 1, cv.CV_32FC3, queryPoints.flat());
			const mat2 = cv.matFromArray(trainPoints.length, 1, cv.CV_32FC2, trainPoints.flat());

			const f = trainImage.image.cols/2/(Math.tan(angle/2*Math.PI/180))
	
			const _mtx = [
				f, 0, trainImage.image.cols / 2,
				0, f, trainImage.image.rows / 2 ,
				0, 0, 1
			]

			const _dist = [ 0, 0, 0, 0 ]

			const mtx = cv.matFromArray(3, 3, cv.CV_64F, _mtx)
			const dist = cv.matFromArray(1, _dist.length, cv.CV_64F, _dist)

			const rvec = new cv.Mat()
			const tvec = new cv.Mat()

			const inliers = new cv.Mat()
			cv.solvePnPRansac(mat1, mat2, mtx, dist, rvec, tvec, false, 100, 8.0, 0.99, inliers)
	
			if(inliers.rows / mat1.rows > 0.8){
				const projectionMatrix = getProjectionMatrix(rvec, tvec, mtx)

				const storePoints = { trainPoints: [], queryPoints: [] }
				for(let i = 0; i < inliers.rows; i++){
					storePoints.trainPoints.push(trainPoints[inliers.intAt(i, 0)])
					storePoints.queryPoints.push(queryPoints[inliers.intAt(i, 0)])
				}
				lastFramesData[sourceImage.id] = storePoints

				mtx.delete()
				dist.delete()

				const _axis = [
					0, 0, 0, 1,
					30, 0, 0, 1,
					0, 30, 0, 1,
					0, 0, -30, 1
				]
				const axis = cv.matFromArray(4, 4, cv.CV_64F, _axis).t()

				const points = dot(projectionMatrix, axis).t()

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

				for(let point of storePoints.trainPoints)
					cv.circle(finalImage, { x: point[0], y: point[1] }, 2, [ 0, 0, 255, 255 ])
				
				projectionMatrix.delete()
				axis.delete()
				points.delete()
			}else{
				lastFramesData[sourceImage.id] = null
			}

			mat1.delete()
			mat2.delete()
		}
		
		if(lastFramesData[sourceImage.id]){
			if(lastFramesData[sourceImage.id].image)
				lastFramesData[sourceImage.id].image.delete()
			lastFramesData[sourceImage.id].image = new cv.Mat(trainImage.image)
		}

		trainImage.delete()
		return imageDataFromMat(finalImage)
	},

	getCameraPosition: async (imageData) => {
		
	}
}

function dot(a, b){
	const res = new cv.Mat
	const zeros = cv.Mat.zeros(a.cols, b.rows, cv.CV_64F)
	cv.gemm(a, b, 1, zeros, 0, res)
	zeros.delete()
	
	return res
}

function getProjectionMatrix(rvec, tvec, mtx){
	const rotationMatrix = new cv.Mat()
	cv.Rodrigues(rvec, rotationMatrix)

	const extrinsicMatrix = new cv.Mat(3, 4, cv.CV_64F)

	for(let i = 0; i < 3; i++){
		for(let j = 0; j < 3; j++){
			extrinsicMatrix.doublePtr(i, j)[0] = rotationMatrix.doubleAt(i, j)
		}
		extrinsicMatrix.doublePtr(i, 3)[0] = tvec.doubleAt(i, 0)
	}

	
	const projectionMatrix = dot(mtx, extrinsicMatrix)

	extrinsicMatrix.delete()
	rotationMatrix.delete()

	return projectionMatrix
}

function getImageKeypoints(image){
	const imgGray = new cv.Mat()
	cv.cvtColor(image, imgGray, cv.COLOR_BGR2GRAY)

	const keypoints = new cv.KeyPointVector()             //Ключевые точки
	const none = new cv.Mat() 
	const descriptors = new cv.Mat()                              //Дескрипторы точек (т.е. некое уникальное значение)

	orb.detectAndCompute(imgGray, none, keypoints, descriptors)

	none.delete()
	image.delete()

	const dispose = () => {
		keypoints.delete()
		descriptors.delete()
		imgGray.delete()
	}

	return { image: imgGray, keypoints, descriptors, delete: dispose }
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