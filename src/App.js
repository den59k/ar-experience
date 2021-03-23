import { useEffect, useRef, useState } from 'react';
import CV from 'services/cv'
import { delay } from 'services/delay'

function App() {

	const sourceVideoRef = useRef()
	const sourceImageRef = useRef()
	const canvasRef = useRef()

	const [ fps, setFps ] = useState(0)

	useEffect(() => {
		
		function initCanvas (width, height){

			const canvas = document.createElement('canvas')
			canvas.width = width
			canvas.height = height
			const ctx = canvas.getContext('2d')

			return (videoElement) => {
				ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height)
				return ctx.getImageData(0, 0, canvas.width, canvas.height)
			}
		}

		async function initCamera(maxVideoSize){

			const stream = await navigator.mediaDevices.getUserMedia({
				audio: false,
				video: {
					facingMode: 'environment',
					width: maxVideoSize,
					height: maxVideoSize,
				},
			})

			sourceVideoRef.current.srcObject = stream
			sourceVideoRef.current.play()

			await delay(250)			//Верь мне, так нужно

			return stream
		}

		//Отдельная функция для инициализации OpenCV
		async function initCV (){
			const res = await CV.init()

			return res.status
		}

		//Функция для загрузки изображения
		async function loadImage(){
			const getImageData = initCanvas(sourceImageRef.current.naturalWidth, sourceImageRef.current.naturalHeight)
			const imageData = getImageData(sourceImageRef.current)
			const res = await CV.loadSourceImage(imageData)
			return res
		}

		//И потом мы объединяем их в одну цепочку
		async function init(){
			const status = await initCV()
			const image = await loadImage()

			return { status, image }
		}
		

		Promise.all([ init(), initCamera(600) ]).then(values => {
			const { image } = values[0]
			console.log(image)

			const getImageData = initCanvas(sourceVideoRef.current.videoWidth, sourceVideoRef.current.videoHeight)
			
			//Мы запускаем цикл, в котором просто будем выводить
			async function computeImage() {

				const imageData = getImageData(sourceVideoRef.current)
				
				const time = performance.now()
				const resultImageData = await CV.estimateCameraPosition(image.id, imageData)
				setFps(Math.round(1000 / (performance.now() - time)))

				canvasRef.current.width = resultImageData.width
				canvasRef.current.height = resultImageData.height
				const ctx = canvasRef.current.getContext('2d')
				ctx.putImageData(resultImageData, 0, 0)
				
				requestAnimationFrame(computeImage)
			}

			computeImage()
		})

	}, [])

	return (
		<div className="App">
			<div className="fps">{fps} FPS</div>
			<video ref={sourceVideoRef} playsInline={true} ></video>
			<canvas ref={canvasRef}></canvas>
			<img src="/images/box.png" alt="Исходное изображение" style={{display: "none"}} ref={sourceImageRef}/>
		</div>
	);
}

export default App;
