# Lab 6 — Human error review

**Current workflow:** [follow the course GitHub steps](LAB6_GITHUB_WORKFLOW.md). Run `python scripts/review_lab6.py --limit 5` to review and save individual decisions without editing JSON.


**Assistant review available:** [read the short grouped review](LAB6_QUICK_REVIEW.md). All 120 entries now have separate assistant annotations; the human fields below remain unchanged.

These 120 errors come from the supplied course predictions, not our trained classifier. Read each text with its gold and predicted topic. Add a category and note, then confirm it in `artifacts/lab6/human_error_review.json`. The report counts only explicitly confirmed entries.

Categories: label ambiguity; Arabic spelling; dialect/code-switching; entity alignment; truncation; retrieval relevance; preprocessing/serving skew; annotation defect; unexplained model confusion. Do not infer a root cause merely from the wrong label.

## 1. FB-008128

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 2. FB-010648

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 3. FB-006088

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 4. FB-000368

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 5. FB-009328

ألعاب الأطفال في حديقة جدة تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 6. FB-009048

الممر في حديقة الدمام غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 7. FB-007848

ألعأب الأطفال في حديقة الدمام تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 8. FB-004368

ألعاب الأطفال في حديقة حي العليا تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 9. FB-000208

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 10. FB-007008

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 11. FB-009488

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 12. FB-002288

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 13. FB-008728

الري متوقف في حديقة حي النرجس

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 14. FB-001808

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 15. FB-007288

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 16. FB-008848

الممر في حديقة حي النرجس غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 17. FB-011568

الري متوقف في حديقة شارع التحلية

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 18. FB-008448

الري متوقف في حديقة حي النرجس

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 19. FB-005928

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 20. FB-006608

الري متوقف في حديقة جدة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 21. FB-003128

ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 22. FB-004528

الممر في حديقة الدمام غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 23. FB-009968

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 24. FB-010008

ألعاب الأطفال في حديقة جدة تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 25. FB-004168

ألعاب الأطفال في حديقة حي العليا تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 26. FB-001688

الري متوقف في حديقة حي النرجس

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 27. FB-011248

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 28. FB-011808

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 29. FB-006288

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 30. FB-009528

الري متوقف في حديقة جدة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 31. FB-000688

الري متوقف في حديقة الدمام

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 32. FB-006568

الري متوقف في حديقة الدمام

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 33. FB-001328

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 34. FB-000968

الممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 35. FB-008608

ألعأب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 36. FB-004768

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 37. FB-000288

ألعاب الأطفال في حديقة الرياض تحتاج صيانة <NATIONAL_ID>

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 38. FB-003608

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 39. FB-001848

الري متوقف في حديقة شارع التحلية

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 40. FB-002928

الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 41. FB-003728

الممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 42. FB-003048

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 43. FB-011448

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 44. FB-002408

لووووسمحت الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 45. FB-003088

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 46. FB-002528

ألعأب الأطفال في حديقة الدمام تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 47. FB-009368

لووووسمحت ألممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 48. FB-009168

الممر في حديقة حي الياسمين غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 49. FB-004448

الري متوقف في حديقة حي العليا

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 50. FB-009248

الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 51. FB-007208

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 52. FB-005768

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 53. FB-009568

ألعاب الأطفال في حديقة طريق الملك فهد تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 54. FB-003008

الري متوقف في حديقة الدمام  

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 55. FB-011888

الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 56. FB-011128

الري متوقف في حديقة الدمام

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 57. FB-003888

الممر في حديقة حي العليا غير مناسب للكراسي المتحركة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 58. FB-003688

ألعاب الأطفال في حديقة طريق الملك فهد تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 59. FB-007688

ألعاب الأطفال في حديقة جدة تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 60. FB-001008

ألممر في حديقة الدمام غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 61. FB-009288

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة <PHONE>

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 62. FB-002368

الري متوقف في حديقة جدة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 63. FB-000248

ألعأب الأطفال في حديقة الدمام تحتاج صيانة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 64. FB-007408

ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 65. FB-006928

الممر في حديقة حي النرجس غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 66. FB-001208

الري متوقف في حديقة الدمام

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 67. FB-000888

الري متوقف في حديقة شارع التحلية

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 68. FB-010248

الري متوقف في حديقة حي النرجس

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 69. FB-002248

الممر في حديقة الدمام غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 70. FB-007448

الري متوقف في حديقة طريق الملك فهد

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 71. FB-004328

الري متوقف في حديقة حي العليا

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 72. FB-005048

ألعاب الأطفال في حديقة طريق الملك فهد تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 73. FB-009208

الممر في حديقة جدة غير مناسب للكراسي المتحركة  

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 74. FB-007728

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 75. FB-005168

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 76. FB-011288

الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 77. FB-003288

ألري متوقف في حديقة الدمام

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 78. FB-003368

ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة 😡 <PHONE>

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 79. FB-002648

ألعاب الأطفال في حديقة حي النرجس تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 80. FB-011528

الممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 81. FB-006528

الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 82. FB-006168

الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 83. FB-000768

ألعاب الأطفال في حديقة الرياض تحتاج صيانة 😡

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 84. FB-004968

ألعاب الأطفال في حديقة الدمام تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 85. FB-003168

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 86. FB-004568

ألعاب الأطفال في حديقة طريق الملك فهد تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 87. FB-004728

لووووسمحت الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 88. FB-002888

الممر في حديقة حي الياسمين غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 89. FB-000608

الري متوقف في حديقة طريق الملك فهد

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 90. FB-000408

الري متوقف في حديقة حي العليا <PHONE>

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 91. FB-007768

ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 92. FB-004048

ألعأب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 93. FB-002728

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 94. FB-008968

ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 95. FB-001488

ألعاب الأطفال في حديقة الدمام تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 96. FB-000648

الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 97. FB-007168

ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 98. FB-005888

لووووسمحت الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 99. FB-010968

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 100. FB-002688

ألعاب الأطفال في حديقة حي العليا تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 101. FB-009128

ألعاب الأطفال في حديقة حي النرجس تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 102. FB-002768

الممر في حديقة الدمام غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 103. FB-001248

لووووسمحت ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 104. FB-007328

الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 105. FB-010608

الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 106. FB-009008

الممر في حديقة حي النرجس غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 107. FB-005608

الري متوقف في حديقة حي العليا

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 108. FB-000728

ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 109. FB-007248

الممر في حديقة حي النرجس غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 110. FB-003848

ألعاب الأطفال في حديقة حي النرجس تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 111. FB-005288

الري متوقف في حديقة شارع التحلية

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 112. FB-010488

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 113. FB-003448

الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 114. FB-009928

الري متوقف في حديقة الرياض

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 115. FB-009808

ألعاب الأطفال في حديقة الدمام تحتاج صيانة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 116. FB-004648

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 117. FB-005848

الري متوقف في حديقة جدة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 118. FB-011688

لووووسمحت الممر في حديقة الدمام غير مناسب للكراسي المتحركة 😡  

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 119. FB-007888

الممر في حديقة جدة غير مناسب للكراسي المتحركة

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no

## 120. FB-008248

الري متوقف في حديقة حي العليا

Gold: **parks** · Prediction: **roads**

- Category:
- Evidence / note:
- Human confirmed: no
