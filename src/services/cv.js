class CV {
	
	constructor(){
		this.worker = new Worker('/js/cv.worker.js') // load worker
		this.waiting = {}
		
		this.worker.onmessage = (e) => {
			const { action, payload } = e.data
			if(this.waiting[action]){
				this.waiting[action](payload)
				delete this.waiting[action]
			}
		}

		this.worker.onerror = (e) => {
			console.log(e)
		}
	}
	
	//Мы промисифицируем данный метод, чтобы ожидать ответ
	_dispatch = (action, payload) => new Promise ((res, rej) => {
		if(this.waiting[action]) return rej (`Action ${action} is not completed`)

		this.worker.postMessage({ action, payload })
		this.waiting[action] = res
	})


	init() {
		return this._dispatch("init")
	}

	convertToGray (imageData){
		return this._dispatch("convertToGray", imageData)
	}

	loadSourceImage(imageData){
		return this._dispatch("loadSourceImage", imageData) 
	}

	matchPoints (sourceImage, imageData){
		return this._dispatch("matchPoints", { sourceImage, imageData })  
	}

	calculatePnP (sourceImage, imageData){
		return this._dispatch("calculatePnP", { sourceImage, imageData }) 
	}

	estimateCameraPosition (id, imageData){
		return this._dispatch("estimateCameraPosition", { id, imageData }) 
	}

}

export default new CV()