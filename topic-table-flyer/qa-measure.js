/**
 * QA:在 Chromium 里量 A4 单页版式,确认没有溢出,并导出 PDF。
 * 用法:node qa-measure.js   (需要 playwright,浏览器用沙箱预装的 Chromium)
 */
const { chromium } = require("playwright");
const path = require("path");

const HTML = "file://" + path.resolve(__dirname, "topic-table-restaurant-flyer.html");
const PDF = path.resolve(__dirname, "topic-table-restaurant-flyer.pdf");
const CHROME = process.env.CHROME_PATH || "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

(async () => {
  const browser = await chromium.launch({ executablePath: CHROME });

  // 1) A4 版式:内容总高不得超过 297mm(1122.5px @96dpi)
  const page = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
  await page.goto(HTML);
  await page.waitForTimeout(600);
  const m = await page.evaluate(() => {
    const sheet = document.querySelector(".sheet");
    const h = (el) => +el.getBoundingClientRect().height.toFixed(1);
    return {
      sheetH: h(sheet),
      contentH: +[...sheet.children].reduce((a, c) => a + c.getBoundingClientRect().height, 0).toFixed(1),
      top: h(document.querySelector(".band-top")),
      body: h(document.querySelector(".body")),
      bottom: h(document.querySelector(".band-bottom")),
      colL: h(document.querySelector(".col-l")),
      colR: h(document.querySelector(".col-r")),
    };
  });
  const over = +(m.contentH - m.sheetH).toFixed(1);
  console.log(JSON.stringify(m), "| overflow:", over + "px =", (over / 3.7795).toFixed(1) + "mm");
  if (over > 0) console.error("!! 内容溢出一页,请调小字号或间距");
  await page.pdf({ path: PDF, format: "A4", printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 } });
  await page.close();

  // 2) 响应式:820px 以下单栏,不得出现横向滚动
  for (const width of [390, 760]) {
    const p = await browser.newPage({ viewport: { width, height: 900 } });
    await p.goto(HTML);
    await p.waitForTimeout(400);
    const r = await p.evaluate(() => ({
      doc: document.documentElement.scrollWidth,
      client: document.documentElement.clientWidth,
    }));
    console.log(`${width}px:`, JSON.stringify(r), r.doc <= r.client + 1 ? "OK" : "!! 横向溢出");
    await p.close();
  }

  await browser.close();
})();
