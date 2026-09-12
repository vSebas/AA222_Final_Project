/**
 * 上海「话题餐桌」— 主理人招募单页(A4 竖版 · 单页 · 左右两栏版)
 * 生成 flyer.pptx,用于上传 Google Drive 转成 Google Slides 编辑。
 * 版式与文案与 topic-table-restaurant-flyer.html 完全一致。
 */
const PptxGenJS = require("pptxgenjs");

/* ---------------- 设计系统 ---------------- */
const C = {
  deep: "1E3227",   // 主色 深墨绿
  paper: "F4F2EC",  // 纸色
  card: "E9E7DE",   // 卡片底
  gold: "C79A3E",   // 唯一强调色
  ink: "17231C",    // 墨
  moss: "4A5F4E",   // 苔绿
  onDeep: "D6D9D1", // 深底上的次级正文
};
const SERIF = "Noto Serif SC";
const SANS = "Noto Sans SC";

const W = 8.27, H = 11.69;   // A4 竖版
const M = 0.6;               // 页边距
const CW = W - 2 * M;        // 内容宽 7.07
const LX = M, LW = 4.45;     // 左栏
const RX = 5.35, RW = 2.32;  // 右栏

const pres = new PptxGenJS();
pres.defineLayout({ name: "A4P", width: W, height: H });
pres.layout = "A4P";
pres.author = "话题餐桌";
pres.title = "上海首店 · 诚邀餐饮主理人合伙";

const s = pres.addSlide();
s.background = { color: C.paper };

/* ---------------- helpers ---------------- */
const T = (text, o) =>
  s.addText(text, Object.assign({ isTextBox: true, valign: "top", margin: 0, fontFace: SANS }, o));

const box = (x, y, w, h, fill) =>
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 } });

// 俯视六人圆桌:金色细圆环 + 六个座位点 + 圆心短文字
function roundTable(cx, cy, R, centerText, centerColor, opt = {}) {
  const dot = opt.dot || R * 0.2;
  s.addShape(pres.ShapeType.ellipse, {
    x: cx - R, y: cy - R, w: R * 2, h: R * 2,
    fill: { type: "none" }, line: { color: C.gold, width: opt.ring || 0.75 },
  });
  [[0, -1], [0.866, -0.5], [0.866, 0.5], [0, 1], [-0.866, 0.5], [-0.866, -0.5]].forEach(([ux, uy]) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: cx + ux * R - dot / 2, y: cy + uy * R - dot / 2, w: dot, h: dot,
      fill: { color: C.gold }, line: { color: C.gold, width: 0.25 },
    });
  });
  const fs = opt.fs || 8;
  T(centerText, {
    x: cx - R * 0.92, y: cy - fs / 144, w: R * 1.84, h: fs / 72 + 0.06,
    fontFace: SERIF, fontSize: fs, color: centerColor, align: "center",
  });
}

/* ================= 页首标题带(深底通栏) ================= */
const TOP_H = 2.45;
box(0, 0, W, TOP_H, C.deep);

T("上海首店 · 诚邀餐饮主理人合伙", {
  x: M, y: 0.55, w: CW, h: 0.17, fontSize: 8, color: C.gold, charSpacing: 2.4,
});
T([{ text: "不是一家新的餐厅,", options: { breakLine: true } }, { text: "而是一种新的餐厅." }], {
  x: M, y: 0.86, w: 5.0, h: 0.86, fontFace: SERIF, fontSize: 26, bold: true,
  color: C.paper, lineSpacingMultiple: 1.14,
});
s.addShape(pres.ShapeType.rect, { x: M, y: 2.07, w: 0.55, h: 0.012, fill: { color: C.gold }, line: { color: C.gold, width: 0 } });
T("上海 · 2026", { x: M + 0.72, y: 2.005, w: 2.0, h: 0.17, fontSize: 8, color: C.onDeep, charSpacing: 1.4 });

roundTable(6.95, 1.22, 0.48, "六人一桌", C.paper, { fs: 8, dot: 0.105 });

/* ================= 左栏 ================= */
T("你好,同在上海的你!", {
  x: LX, y: 2.80, w: LW, h: 0.34, fontFace: SERIF, fontSize: 16.5, bold: true, color: C.deep,
});
T([{ text: "以兴趣话题为纽带,连接城市里的个人。", options: { breakLine: true } },
   { text: "告别屏幕,重建真诚有趣的面对面陌生人社交。" }], {
  x: LX, y: 3.24, w: LW, h: 0.50, fontSize: 10.2, color: C.moss, lineSpacingMultiple: 1.3,
});

T("这对门店意味着什么", { x: LX, y: 4.28, w: LW, h: 0.30, fontFace: SERIF, fontSize: 14, bold: true, color: C.deep });
[["固定的出餐结构", 0, 0], ["可预测的上座情况", 1, 0], ["以内容为导向的客流", 0, 1], ["可扩张的商业模式", 1, 1]]
  .forEach(([txt, col, row]) => {
    const x = LX + col * 2.32, y = 4.76 + row * 0.42;
    s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.065, w: 0.085, h: 0.085, fill: { color: C.gold }, line: { color: C.gold, width: 0 } });
    T(txt, { x: x + 0.19, y, w: 2.05, h: 0.22, fontSize: 10.8, bold: true, color: C.deep });
  });

T("我们在找的人", { x: LX, y: 5.98, w: LW, h: 0.30, fontFace: SERIF, fontSize: 14, bold: true, color: C.deep });
T("一位有实际开店经验的餐饮主理人,与我们共同落地第一家店。您负责餐饮的基础设施,我们负责将这家餐厅变得独一无二。", {
  x: LX, y: 6.34, w: LW, h: 0.50, fontSize: 9.2, color: C.moss, lineSpacingMultiple: 1.3,
});

[["门店", "主理人负责", ["餐厅资质", "厨师团队", "餐食供应"]],
 ["场景", "我们负责", ["品牌、内容与传播", "话题策划与选座系统", "预订、会员与社群运营", "空间叙事"]]]
  .forEach(([tag, title, list], i) => {
    const x = LX + i * 2.33;
    box(x, 6.94, 2.12, 1.66, C.card);
    T(tag, { x: x + 0.19, y: 7.08, w: 1.4, h: 0.16, fontSize: 7.5, color: C.gold, charSpacing: 2 });
    T(title, { x: x + 0.19, y: 7.27, w: 1.7, h: 0.24, fontFace: SERIF, fontSize: 11.5, bold: true, color: C.deep });
    list.forEach((li, j) => {
      T(li, { x: x + 0.19, y: 7.62 + j * 0.235, w: 1.78, h: 0.21, fontSize: 9, color: C.moss, bullet: { indent: 10 } });
    });
  });

T([{ text: "第一步,只是坐下来聊一小时:", options: { breakLine: true } },
   { text: "不谈合同,先谈这家店该长什么样。" }], {
  x: LX, y: 10.28, w: LW, h: 0.56, fontFace: SERIF, fontSize: 13.5, color: C.deep, lineSpacingMultiple: 1.24,
});

/* ================= 右栏:发起人 + 微信二维码 ================= */
T("关于发起人", { x: RX, y: 2.80, w: RW, h: 0.30, fontFace: SERIF, fontSize: 14, bold: true, color: C.deep });

// 圆形头像卡。拿到照片后把 ［头像］ 文本框换成 addImage({ rounding: true }) 即可。
[["［姓名］", "哈佛大学", "产品与工程", 3.22],
 ["［姓名］", "斯坦福大学", "品牌与内容", 4.68]].forEach(([name, school, bio, y]) => {
  box(RX, y, RW, 1.30, C.card);
  const cx = RX + 0.17 + 0.44, cy = y + 0.65;
  s.addShape(pres.ShapeType.ellipse, {
    x: cx - 0.44, y: cy - 0.44, w: 0.88, h: 0.88,
    fill: { color: C.paper }, line: { color: C.gold, width: 0.75 },
  });
  T("［头像］", { x: cx - 0.40, y: cy - 0.07, w: 0.80, h: 0.16, fontSize: 6.5, color: C.moss, align: "center" });
  T(name, { x: RX + 1.14, y: y + 0.32, w: 1.02, h: 0.24, fontFace: SERIF, fontSize: 12, bold: true, color: C.deep });
  T(school, { x: RX + 1.14, y: y + 0.60, w: 1.02, h: 0.16, fontSize: 7.4, color: C.gold, charSpacing: 1.6 });
  T(bio, { x: RX + 1.14, y: y + 0.80, w: 1.02, h: 0.20, fontSize: 8.8, color: C.moss });
});

T("开店的事我们没做过,也不打算假装做过。我们能带来的是内容、话题,和把人聚到同一张桌子上的方法。剩下的,需要一位真正懂店的人。", {
  x: RX, y: 6.14, w: RW, h: 0.90, fontSize: 8.8, color: C.moss, lineSpacingMultiple: 1.34,
});

// 联系卡:小标签 → 二维码 → 微信/邮箱
box(RX, 9.05, RW, 1.90, C.card);
T("见面", { x: RX + 0.19, y: 9.19, w: 1.2, h: 0.16, fontSize: 7.5, color: C.gold, charSpacing: 2 });
s.addShape(pres.ShapeType.rect, {
  x: RX + (RW - 1.05) / 2, y: 9.42, w: 1.05, h: 1.05,
  fill: { color: C.paper }, line: { color: C.gold, width: 0.75 },
});
T([{ text: "［微信", options: { breakLine: true } }, { text: "二维码］" }], {
  x: RX + (RW - 1.05) / 2, y: 9.83, w: 1.05, h: 0.34, fontSize: 6.5, color: C.moss, align: "center", lineSpacingMultiple: 1.2,
});
[["微信", "［填入微信号］", 10.55], ["邮箱", "［填入邮箱］", 10.74]].forEach(([k, v, y]) => {
  T(k, { x: RX + 0.19, y, w: 0.42, h: 0.18, fontSize: 8.8, color: C.deep });
  T(v, { x: RX + 0.66, y, w: 1.5, h: 0.18, fontSize: 8.8, color: C.moss });
});

/* ================= 页尾细带 ================= */
const BOT_Y = 11.15;
box(0, BOT_Y, W, H - BOT_Y, C.deep);
T("上海首店 · 诚邀餐饮主理人合伙", { x: M, y: 11.32, w: 4.5, h: 0.17, fontSize: 7.5, color: C.gold, charSpacing: 2.2 });
T("上海 · 可线下面谈", { x: 5.0, y: 11.32, w: 2.67, h: 0.17, fontSize: 7.5, color: C.onDeep, charSpacing: 1.2, align: "right" });

/* ---------------- 演讲备注 ---------------- */
s.addNotes(
  "A4 竖版单页招募单(左右两栏版)。需要发起人自行替换的内容:\n" +
  "1) 右栏两张头像卡的［头像］与［姓名］—— 换成圆形照片和真名;\n" +
  "2) 联系卡的［微信二维码］［填入微信号］［填入邮箱］;\n" +
  "3) 品牌名(目前全页未出现,可加在页首小字或页尾)。\n" +
  "配色:深墨绿 1E3227 / 纸色 F4F2EC / 卡片 E9E7DE / 金 C79A3E。字体只用思源宋体与思源黑体。"
);

pres.writeFile({ fileName: process.argv[2] || "flyer.pptx" }).then(f => console.log("written:", f));
