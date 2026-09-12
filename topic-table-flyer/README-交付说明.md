# 主理人招募单页 — 交付说明

同一张 **A4 竖版单页**(210×297mm)的两种格式,版式、文案、分区顺序完全一致。

| 文件 | 用途 |
|---|---|
| `topic-table-restaurant-flyer.html` | A · 单文件 HTML,内联 CSS,离线可开。Ctrl+P 直接出满版 A4 PDF(已验证恰好一页) |
| `topic-table-restaurant-flyer.pdf` | A 的导出结果,邮件附件 / 打印用 |
| `flyer-min.pptx` | B · A4 竖版单页 pptx(已 deflate 压缩,20KB) |
| `build.js` | 生成 `flyer.pptx` 的 pptxgenjs 脚本 |

Google Slides 版(可直接在线编辑)已上传至 Drive,见对话中的链接。

## 需要发起人自己替换的地方

**联系方式(占位符,保持方括号形态,勿编造)**

- `［填入微信号］` — 页尾右侧联系卡
- `［填入邮箱］` — 同上

在 Google Slides 里双击那两行文字直接改即可;HTML 版搜索 `［填入` 替换。

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
