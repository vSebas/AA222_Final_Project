# 主理人招募单页 — 交付说明

同一张 **A4 竖版单页**(210×297mm)的两种格式,版式、文案、分区顺序完全一致。

| 文件 | 用途 |
|---|---|
| `topic-table-restaurant-flyer.html` | A · 单文件 HTML,内联 CSS,离线可开。Ctrl+P 直接出满版 A4 PDF(已验证恰好一页) |
| `topic-table-restaurant-flyer.pdf` | A 的导出结果,邮件附件 / 打印用 |
| `flyer-min.pptx` | B · A4 竖版单页 pptx(已 deflate 压缩,20KB) |
| `build.js` | 生成 `flyer.pptx` 的 pptxgenjs 脚本 |

Google Slides 版(可直接在线编辑)已上传至 Drive,见对话中的链接。

## 版式(v2 · 左右两栏)

- 页首深底带:小字 + 主标题 + 圆桌图形 + 金线「上海 · 2026」
- 左栏:问候语与引言 / 这对门店意味着什么(四条) / 我们在找的人(引言 + 两张分工卡) / 见面邀约
- 右栏:关于发起人(两张圆形头像卡,上下排列)+ 坦白段 + 微信二维码联系卡
- 页尾深底带压到 4.4mm 的一条细边

## 需要发起人自己替换的地方

**图片(目前是占位框)**

- 两张圆形头像 — 右栏 `［头像］`。HTML 版把 `<div class="avatar">［头像］</div>` 换成
  `<div class="avatar"><img src="photo.jpg" alt=""></div>` 即可(已写好圆形裁切样式)。
  pptx 版用 `addImage({ rounding: true })` 替换对应文本框。
- 微信二维码 — `［微信二维码］` 占位方框,同样换成 `<img>`。

**文字占位符(保持方括号形态,勿编造)**

- `［姓名］` ×2 — 右栏两张头像卡
- `［填入微信号］` `［填入邮箱］` — 右栏联系卡

**目前整页还没有品牌名** —— 这是最大的缺口。定下来之后建议加在页首小字(`上海首店 · 诚邀餐饮主理人合伙`)或页尾。

**以下内容刻意未写进材料,面谈时再谈**(brief 第 6 节):
菜系方向与客单价区间 / 目标商圈与面积量级 / 股权结构的可谈范围(现在只写“面谈”)/ 首店时间表。

## 设计系统(改动时请沿用)

| 用途 | Hex |
|---|---|
| 主色 深墨绿 | `1E3227` |
| 纸色 | `F4F2EC` |
| 卡片底 | `E9E7DE` |
| 金(唯一强调色) | `C79A3E` |
| 墨 | `17231C` |
| 苔绿 | `4A5F4E` |

字体只用两种:标题 思源宋体 / `Noto Serif SC`,正文 思源黑体 / `Noto Sans SC`。
视觉母题只有一个:俯视的六人圆桌(金色细圆环 + 六个座位点 + 圆心短文字),全页出现三次。

## 重新生成 pptx

```bash
npm install pptxgenjs
node build.js flyer.pptx
python3 -c "
import zipfile
src=zipfile.ZipFile('flyer.pptx')
out=zipfile.ZipFile('flyer-min.pptx','w',zipfile.ZIP_DEFLATED,compresslevel=9)
for i in src.infolist(): out.writestr(i.filename, src.read(i.filename))
out.close()"
```

文案为定稿文案,请勿改写。
