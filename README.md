## Компиляция OpenCV

Для корректной работы на десктопе и на мобильных устройствах хорошо работает связка 1.40.1 и OpenCV 3.4.13

Билд производится такой командой (CentOS 8)
```
emcmake python3 ./opencv/platforms/js/build_js.py build_wasm --build_wasm --build_flags="-s USE_PTHREADS=0 -O3"
```