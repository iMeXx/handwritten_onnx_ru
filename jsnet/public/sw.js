// Service Worker: офлайн-кэш приложения jsnet (регистратор — /js/lines.js со
// страницы распознавания). Правило простое: /api/* — всегда сеть (данные
// автора не кэшируем); HTML — сеть, при отсутствии сети отдаём кэш;
// всё остальное (ort-wasm, модель .onnx/.ort ~18 МБ, line_vocab.json,
// terms.json, скрипты) — кэш-first: второй и последующие заходы на планшете
// не качают модель заново.
// На http://IP (локальная сеть, не Secure Context) service worker недоступен —
// регистрация молча не срабатывает, страница работает без кэша.
// При обновлении модели/словаря/ort — поднимите версию CACHE (старые кэши
// вычищаются в activate). v2 — новая растровая модель (line_int8.onnx/.ort);
// v3 — статус в тулбаре /train.html; v6 — векторное направление закрыто,
// воркер/препроцессинг/lines.js откатаны к растровому срезу (cache-first
// отдал бы старые гибридные копии).
const CACHE = "inknet-jsnet-v7"; // v7: beamDecode oovPen (словарное давление)

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

const put = (req, res) =>
  caches.open(CACHE).then((c) => c.put(req, res.clone()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return; // API — только живая сеть

  const isPage = url.pathname === "/" || url.pathname.endsWith(".html");
  if (isPage) {
    e.respondWith(
      fetch(req)
        .then((res) => { put(req, res); return res; })
        .catch(() => caches.match(req).then((hit) => hit || Response.error()))
    );
    return;
  }

  e.respondWith(
    caches.match(req).then(
      (hit) =>
        hit ||
        fetch(req).then((res) => {
          if (res.ok) put(req, res);
          return res;
        })
    )
  );
});
