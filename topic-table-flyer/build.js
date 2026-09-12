/**
 * 上海「话题餐桌」— 主理人招募单页(A4 竖版 · 单页)
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

const W = 8.27, H = 11.69;      // A4 竖版
const M = 0.6;                  // 页边距
const CW = W - 2 * M;           // 内容宽 7.07

const pres = new PptxGenJS();
pres.defineLayout({ name: "A4P", width: W, height: H });
pres.layout = "A4P";
pres.author = "话题餐桌";
pres.title = "上海首店 · 诚邀餐饮主理人合伙";

const s = pres.addSlide();
s.background = { color: C.paper };

/* ---------------- helpers ---------------- */
// 所有 addText 统一 valign:top / isTextBox / margin:0(见 brief 5.2)
const T = (text, o) =>
  s.addText(text, Object.assign({ isTextBox: true, valign: "top", margin: 0, fontFace: SANS }, o));

// 俯视六人圆桌:金色细圆环 + 六个金色实心座位点 + 圆心短文字
function roundTable(cx, cy, R, centerText, centerColor, opt = {}) {
  const dot = opt.dot || R * 0.2;
  s.addShape(pres.ShapeType.ellipse, {
    x: cx - R, y: cy - R, w: R * 2, h: R * 2,
    fill: { type: "none" }, line: { color: C.gold, width: opt.ring || 0.75 },
  });
  const seats = [[0, -1], [0.866, -0.5], [0.866, 0.5], [0, 1], [-0.866, 0.5], [-0.866, -0.5]];
  seats.forEach(([ux, uy]) => {
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

/* ================= 区块一 / 页首标题带(深底通栏) ================= */
const TOP_H = 2.40;
s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: W, h: TOP_H, fill: { color: C.deep }, line: { color: C.deep, width: 0 } });

T("上海首店 · 诚邀餐饮主理人合伙", {
  x: M, y: 0.50, w: CW, h: 0.17, fontSize: 8, color: C.gold, charSpacing: 2.4,
});
T([{ text: "不是一家新的餐厅,", options: { breakLine: true } }, { text: "而是一种新的餐厅" }], {
  x: M, y: 0.71, w: 5.1, h: 0.90, fontFace: SERIF, fontSize: 28, bold: true,
  color: C.paper, lineSpacingMultiple: 1.12,
});
T("一张桌子,六个人,一个当晚的话题。我们把一顿饭重新组织了一遍。", {
  x: M, y: 1.86, w: 4.9, h: 0.21, fontSize: 9.5, color: C.onDeep,
});
s.addShape(pres.ShapeType.rect, { x: M, y: 2.16, w: 0.55, h: 0.012, fill: { color: C.gold }, line: { color: C.gold, width: 0 } });
T("上海 · 2026", { x: M + 0.72, y: 2.095, w: 2.0, h: 0.17, fontSize: 8, color: C.onDeep, charSpacing: 1.4 });

roundTable(6.92, 1.22, 0.52, "六人一桌", C.paper, { fs: 8.5, dot: 0.115 });

/* ================= 区块二 / 我们在做什么(纸色) ================= */
T("我们在做什么", { x: M, y: 2.64, w: 4.0, h: 0.26, fontFace: SERIF, fontSize: 13.5, bold: true, color: C.deep });
T([{ text: "城市里的年轻人不缺饭局,", options: { breakLine: true } }, { text: "缺的是一场能认真说话的饭。" }], {
  x: M, y: 2.96, w: 4.9, h: 0.48, fontFace: SERIF, fontSize: 12, color: C.deep, lineSpacingMultiple: 1.20,
});

const POINTS = [
  ["一桌一个话题", "每周的话题由我们策划、提前公布。客人为了谈什么而来。"],
  ["以座位为单位", "六人一桌,单座预订。一个人也可以来。"],
  ["常客长在关系上", "在一张桌上谈得投机的人,会为了下一场再来。"],
];
POINTS.forEach(([h, d], i) => {
  const y = 3.53 + i * 0.44;
  T(h, { x: M, y, w: 4.9, h: 0.2, fontSize: 10, bold: true, color: C.deep });
  T(d, { x: M, y: y + 0.205, w: 4.9, h: 0.19, fontSize: 9, color: C.moss });
});

// 右侧卡片
s.addShape(pres.ShapeType.rect, { x: 5.72, y: 2.89, w: 1.95, h: 1.86, fill: { color: C.card }, line: { color: C.card, width: 0 } });
roundTable(6.695, 3.42, 0.45, "今晚的话题", C.deep, { fs: 7.5, dot: 0.1 });
T("话题提前公布,客人按兴趣选座。", { x: 5.92, y: 4.21, w: 1.58, h: 0.4, fontSize: 8, color: C.moss, lineSpacingMultiple: 1.2 });

/* ================= 区块三 / 这对门店意味着什么(纸色) ================= */
T("这对门店意味着什么", { x: M, y: 5.05, w: 4.5, h: 0.26, fontFace: SERIF, fontSize: 13.5, bold: true, color: C.deep });
T("一个正常开店的人会先问的四件事。", { x: M, y: 5.37, w: CW, h: 0.19, fontSize: 9, color: C.moss });

const FOUR = [
  ["出餐结构是固定的", "菜单收得很窄,后厨不必应付一张长菜单。"],
  ["上座是可预测的", "按座位预订,当晚坐多少人在开门前就知道。"],
  ["客流由内容带来", "话题和社群由我们运营,门店不必自己去买流量。"],
  ["跑通了可以复制", "首店验证之后,同一套模型能开第二家、第三家。"],
];
FOUR.forEach(([h, d], i) => {
  const col = i % 2, row = (i - col) / 2;
  const x = M + col * 3.62, y = 5.69 + row * 0.48;
  s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.058, w: 0.085, h: 0.085, fill: { color: C.gold }, line: { color: C.gold, width: 0 } });
  T(h, { x: x + 0.18, y, w: 3.25, h: 0.2, fontSize: 10, bold: true, color: C.deep });
  T(d, { x: x + 0.18, y: y + 0.205, w: 3.25, h: 0.19, fontSize: 9, color: C.moss });
});

/* ================= 区块四 / 我们在找的人(纸色) ================= */
T("我们在找的人", { x: M, y: 6.81, w: 4.0, h: 0.26, fontFace: SERIF, fontSize: 13.5, bold: true, color: C.deep });
T("一位有实际开店经验的餐饮主理人,与我们共同落地第一家店。开店的专业交给你,我们负责把对的人和对的话题带进来。", {
  x: M, y: 7.13, w: CW, h: 0.38, fontSize: 9, color: C.moss, lineSpacingMultiple: 1.28,
});

const CARDS = [
  ["门店", "主理人负责", ["选址、筹建与证照", "后厨、出品与菜单落地", "成本结构与供应链", "日常运营与团队管理"]],
  ["场景", "我们负责", ["品牌、内容与传播", "话题策划与选座系统", "预订、会员与社群运营", "空间叙事与融资"]],
];
CARDS.forEach(([tag, title, list], i) => {
  const x = M + i * 3.67;
  s.addShape(pres.ShapeType.rect, { x, y: 7.59, w: 3.40, h: 1.04, fill: { color: C.card }, line: { color: C.card, width: 0 } });
  T(tag, { x: x + 0.2, y: 7.72, w: 1.4, h: 0.16, fontSize: 7.5, color: C.gold, charSpacing: 2 });
  T(title, { x: x + 0.2, y: 7.90, w: 2.6, h: 0.23, fontFace: SERIF, fontSize: 11.5, bold: true, color: C.deep });
  // 四条并列,卡内分两列,避免单页溢出
  list.forEach((li, j) => {
    const c = j % 2, r = (j - c) / 2;
    T(li, {
      x: x + 0.2 + c * 1.56, y: 8.17 + r * 0.20, w: 1.52, h: 0.19,
      fontSize: 8.5, color: C.moss, bullet: { indent: 10 },
    });
  });
});
T("合作方式:股权合伙,具体结构面谈。第一家店共同署名,验证跑通后按同一模型复制。", {
  x: M, y: 8.75, w: CW, h: 0.19, fontSize: 8.5, color: C.moss,
});

/* ================= 区块五 / 关于发起人 + 见面邀约(页尾深底带) ================= */
const BOT_Y = 9.18;
s.addShape(pres.ShapeType.rect, { x: 0, y: BOT_Y, w: W, h: H - BOT_Y, fill: { color: C.deep }, line: { color: C.deep, width: 0 } });

T("关于发起人", { x: M, y: 9.40, w: 4.0, h: 0.26, fontFace: SERIF, fontSize: 13.5, bold: true, color: C.paper });
T("两位毕业于哈佛大学与斯坦福大学的发起人。一人来自产品与工程,一人来自品牌与内容,过去几年做的都是与“人如何相遇”有关的事。", {
  x: M, y: 9.72, w: 4.78, h: 0.38, fontSize: 8.5, color: C.onDeep, lineSpacingMultiple: 1.28,
});
T("开店的事我们没做过,也不打算假装做过。我们能带来的是内容、话题,和把人聚到同一张桌子上的方法。剩下的,需要一位真正懂店的人。", {
  x: M, y: 10.14, w: 4.78, h: 0.38, fontSize: 8.5, color: C.onDeep, lineSpacingMultiple: 1.28,
});
T([{ text: "第一步,只是坐下来聊一小时:", options: { breakLine: true } }, { text: "不谈合同,先谈这家店该长什么样。" }], {
  x: M, y: 10.56, w: 4.78, h: 0.42, fontFace: SERIF, fontSize: 11.5, color: C.paper, lineSpacingMultiple: 1.20,
});
T("上海首店 · 诚邀餐饮主理人合伙", { x: M, y: 11.06, w: 5.0, h: 0.17, fontSize: 7.5, color: C.gold, charSpacing: 2.2 });

// 右侧联系卡:金色细描边,无填充
s.addShape(pres.ShapeType.rect, {
  x: 5.55, y: 9.38, w: 2.12, h: 1.80, fill: { type: "none" }, line: { color: C.gold, width: 0.75, transparency: 45 },
});
roundTable(6.61, 9.85, 0.32, "见面", C.paper, { fs: 6.5, dot: 0.075 });
[["微信", "［填入微信号］"], ["邮箱", "［填入邮箱］"], ["地点", "上海 · 可线下面谈"]].forEach(([k, v], i) => {
  const y = 10.34 + i * 0.23;
  T(k, { x: 5.74, y, w: 0.45, h: 0.18, fontSize: 8, color: C.gold });
  T(v, { x: 6.24, y, w: 1.32, h: 0.18, fontSize: 8, color: C.onDeep });
});

/* ---------------- 演讲备注 ---------------- */
s.addNotes(
  "A4 竖版单页招募单。需要发起人自行替换的内容:\n" +
  "1) 右下角联系卡的［填入微信号］［填入邮箱］;\n" +
  "2) 品牌名(目前全页未出现,可加在页首小字或页尾);\n" +
  "3) 菜系方向与客单价、目标商圈/面积、股权结构可谈范围、首店时间表 —— 这些当前刻意未写,面谈时再谈。\n" +
  "文案为定稿文案,请勿改写。配色:深墨绿 1E3227 / 纸色 F4F2EC / 卡片 E9E7DE / 金 C79A3E。"
);

pres.writeFile({ fileName: process.argv[2] || "flyer.pptx" }).then(f => console.log("written:", f));
