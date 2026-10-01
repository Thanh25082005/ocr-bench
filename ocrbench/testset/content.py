"""Sinh nội dung giả nhưng hợp lý cho tài liệu tổng hợp (tiếng Ả Rập / tiếng Anh / trộn).

Mọi hàm nhận một `random.Random` để kết quả lặp lại được theo seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

AR_DIGITS = str.maketrans("0123456789.,", "٠١٢٣٤٥٦٧٨٩٫٬")

# --- Từ vựng tiếng Ả Rập ------------------------------------------------------------

AR_MALE = ["محمد", "أحمد", "عبدالله", "خالد", "سعيد", "عمر", "يوسف", "علي", "حسن", "إبراهيم", "فهد", "ناصر",
           "طارق", "سامي", "ماجد", "وليد", "رامي", "هشام", "عادل", "كريم", "منصور", "بدر", "زياد", "حمزة"]
AR_FEMALE = ["فاطمة", "عائشة", "مريم", "نورة", "سارة", "ليلى", "هدى", "ريم", "أمل", "سلمى", "دينا", "رنا",
             "هند", "لينا", "منى", "جميلة", "أسماء", "شيماء"]
AR_FAMILY = ["العتيبي", "القحطاني", "الحربي", "الزهراني", "الشمري", "المصري", "الحسيني", "الخطيب", "النجار",
             "الحداد", "السعدي", "البكري", "العلي", "الأنصاري", "الكبيسي", "الدوسري", "الشهري", "الغامدي",
             "المالكي", "الهاشمي", "الرشيدي", "الجبوري", "التميمي", "العمري"]
AR_CITIES = [("الرياض", "المملكة العربية السعودية"), ("جدة", "المملكة العربية السعودية"),
             ("الدمام", "المملكة العربية السعودية"), ("دبي", "الإمارات العربية المتحدة"),
             ("أبوظبي", "الإمارات العربية المتحدة"), ("الدوحة", "دولة قطر"), ("الكويت", "دولة الكويت"),
             ("المنامة", "مملكة البحرين"), ("مسقط", "سلطنة عمان"), ("عمّان", "المملكة الأردنية الهاشمية"),
             ("القاهرة", "جمهورية مصر العربية"), ("الإسكندرية", "جمهورية مصر العربية"),
             ("الدار البيضاء", "المملكة المغربية"), ("تونس", "الجمهورية التونسية")]
AR_STREETS = ["شارع الملك فهد", "شارع العليا", "طريق الملك عبدالعزيز", "شارع التحلية", "شارع الأمير سلطان",
              "شارع الجامعة", "شارع النيل", "شارع الاستقلال", "طريق المطار", "شارع الكورنيش"]
AR_DISTRICTS = ["حي النخيل", "حي الروضة", "حي السلامة", "حي الملقا", "حي الزهراء", "حي الشاطئ", "حي الياسمين"]
AR_COMPANIES = ["الأفق للتجارة", "النخبة للمقاولات", "الريادة للتقنية", "الخليج للخدمات اللوجستية",
                "الواحة للأغذية", "المستقبل للاستشارات", "البناء الحديث", "النور للإلكترونيات",
                "الفجر للطباعة والنشر", "الأمانة للتأمين", "السلام للتطوير العقاري", "الرواد للأنظمة الذكية"]
AR_COMPANY_TYPES = ["شركة", "مؤسسة", "مجموعة"]
AR_TITLES_JOB = ["المدير العام", "مدير المشتريات", "المدير المالي", "رئيس مجلس الإدارة", "مدير المشاريع",
                 "مدير الموارد البشرية"]
AR_WEEKDAYS = ["الأحد", "الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "السبت"]
AR_ORDINALS = ["الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع", "الثامن", "التاسع", "العاشر"]
AR_CURRENCIES = ["ريال سعودي", "درهم إماراتي", "دينار كويتي", "ريال قطري", "جنيه مصري", "دينار أردني"]
AR_PRODUCTS = ["جهاز حاسوب محمول", "طابعة ليزر", "شاشة عرض مقاس ٢٤ بوصة", "ورق طباعة مقاس A4",
               "حبر طابعة أسود", "كرسي مكتب", "طاولة اجتماعات", "خدمة صيانة سنوية", "اشتراك برنامج محاسبة",
               "كابل شبكة", "جهاز توجيه لاسلكي", "قرص صلب خارجي", "لوحة مفاتيح", "فأرة لاسلكية", "خزانة ملفات",
               "مكيف هواء", "خدمة تركيب", "ساعات استشارة", "رسوم نقل وشحن", "كاميرا مراقبة", "جهاز عرض ضوئي"]
# Tên sản phẩm tiếng Anh xuất hiện trong tài liệu tiếng Ả Rập (trường hợp "trộn")
MIXED_PRODUCTS = ["Dell Latitude 5440", "HP LaserJet Pro M404", "Microsoft 365 Business", "Cisco RV340",
                  "Samsung T7 1TB", "Logitech MX Keys", "Lenovo ThinkPad T14", "Epson EcoTank L3250",
                  "Canon EOS R50", "Windows Server 2022", "Oracle Database SE2", "Hikvision DS-2CD2143"]
AR_PURPOSES = ["توريد وتركيب أجهزة الحاسب الآلي ولوازمها", "تقديم خدمات الصيانة الدورية للمباني",
               "تصميم وتطوير نظام إلكتروني لإدارة الموارد", "تقديم خدمات النقل والتوزيع",
               "توريد المواد الغذائية للمقصف", "تنفيذ أعمال التشطيبات الداخلية للمكاتب"]
AR_CONTRACT_TITLES = ["عقد تقديم خدمات", "اتفاقية توريد", "عقد صيانة", "عقد تنفيذ أعمال", "اتفاقية تعاون"]

AR_CLAUSES = [
    "يعتبر التمهيد السابق جزءاً لا يتجزأ من هذا العقد ويقرأ ويفسر معه.",
    "يلتزم الطرف الثاني بتنفيذ الأعمال المتفق عليها وفقاً للمواصفات الفنية المرفقة بهذا العقد.",
    "مدة هذا العقد {months} شهراً تبدأ من تاريخ توقيعه، ويجوز تجديدها باتفاق الطرفين كتابةً.",
    "يدفع الطرف الأول للطرف الثاني مبلغاً إجمالياً قدره {amount} {currency} على {parts} دفعات متساوية.",
    "في حال تأخر الطرف الثاني عن التسليم يحق للطرف الأول خصم غرامة تأخير بنسبة {pct}% عن كل أسبوع تأخير.",
    "يلتزم الطرفان بالحفاظ على سرية المعلومات والبيانات التي يطلعان عليها بموجب هذا العقد.",
    "لا يجوز لأي من الطرفين التنازل عن هذا العقد أو جزء منه للغير إلا بموافقة كتابية من الطرف الآخر.",
    "يحق لأي من الطرفين إنهاء هذا العقد بإشعار كتابي مدته {days} يوماً في حال إخلال الطرف الآخر بالتزاماته.",
    "تخضع أحكام هذا العقد لأنظمة {country}، وتختص محاكم مدينة {city} بالنظر في أي نزاع ينشأ عنه.",
    "يتحمل الطرف الثاني مسؤولية أي أضرار تلحق بممتلكات الطرف الأول أثناء تنفيذ الأعمال.",
    "تكون جميع المراسلات بين الطرفين على العناوين المذكورة في صدر هذا العقد أو عبر البريد الإلكتروني {email}.",
    "يقدم الطرف الثاني ضماناً على الأعمال المنفذة لمدة {months} شهراً من تاريخ الاستلام النهائي.",
    "لا يحق للطرف الثاني التعاقد من الباطن لتنفيذ أي جزء من الأعمال دون موافقة مسبقة من الطرف الأول.",
    "تحرر هذا العقد من نسختين أصليتين، بيد كل طرف نسخة للعمل بموجبها.",
]
AR_LETTER_SUBJECTS = ["طلب عرض أسعار", "إشعار بتغيير العنوان", "طلب تمديد مدة العقد", "تأكيد استلام الشحنة",
                      "طلب صرف مستحقات مالية", "دعوة لحضور اجتماع"]
AR_LETTER_BODIES = [
    "نود إفادتكم بأنه تم استلام خطابكم رقم {ref} بتاريخ {date}، وقد تمت دراسته من قبل الإدارة المختصة.",
    "نرجو التكرم بتزويدنا بعرض أسعار لتوريد {product} بكمية {qty} وحدة، على أن يشمل العرض مدة التوريد وشروط الدفع.",
    "نحيطكم علماً بأن الاجتماع سيعقد يوم {weekday} الموافق {date} في تمام الساعة {hour} صباحاً بمقر الشركة.",
    "وعليه نأمل منكم التكرم بالاطلاع واتخاذ ما يلزم حيال ذلك في أقرب وقت ممكن.",
    "علماً بأن المبلغ المستحق قدره {amount} {currency} وفقاً للفاتورة المرفقة.",
    "وفي حال وجود أي استفسار يرجى التواصل معنا على الرقم {phone} أو البريد الإلكتروني {email}.",
]
# (tiêu đề biểu mẫu, các lý do phù hợp với biểu mẫu đó)
AR_FORMS = [
    ("نموذج طلب توظيف", ["الرغبة في الانضمام إلى فريق العمل", "اكتساب خبرة مهنية جديدة"]),
    ("نموذج فتح حساب", ["فتح حساب جاري", "فتح حساب توفير"]),
    ("استمارة تسجيل متدرب", ["الالتحاق بالبرنامج التدريبي", "تطوير المهارات الإدارية"]),
    ("نموذج طلب إجازة", ["إجازة سنوية", "ظروف عائلية", "إجازة مرضية"]),
    ("نموذج تحديث بيانات عميل", ["تحديث البيانات الشخصية", "تغيير العنوان ورقم الجوال"]),
]

# --- Từ vựng tiếng Anh ----------------------------------------------------------------

EN_FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
            "William", "Susan", "Daniel", "Sarah", "Thomas", "Karen", "Omar", "Layla", "Ahmed", "Nadia"]
EN_LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Taylor", "Clark",
           "Lewis", "Walker", "Hall", "Young", "Haddad", "Khoury", "Rahman", "Saleh"]
EN_CITIES = [("London", "United Kingdom"), ("Manchester", "United Kingdom"), ("New York", "United States"),
             ("Chicago", "United States"), ("Toronto", "Canada"), ("Sydney", "Australia"), ("Dubai", "United Arab Emirates"),
             ("Riyadh", "Saudi Arabia"), ("Doha", "Qatar"), ("Cairo", "Egypt")]
EN_STREETS = ["King Street", "Market Road", "Station Avenue", "Park Lane", "Harbour Drive", "Queen's Road",
              "Victoria Street", "Elm Street", "Business Bay Boulevard"]
EN_COMPANIES = ["Horizon Trading", "Summit Engineering", "BrightPath Technologies", "Gulf Logistics",
                "Oasis Foods", "Future Consulting", "Modern Build", "Northwind Electronics", "Crescent Printing",
                "Trust Insurance", "Peace Real Estate", "Pioneer Smart Systems"]
EN_COMPANY_SUFFIX = ["Ltd.", "LLC", "Group", "Co.", "Inc."]
EN_TITLES_JOB = ["General Manager", "Procurement Manager", "Chief Financial Officer", "Chairman",
                 "Project Manager", "HR Manager"]
EN_PRODUCTS = ["Laptop computer", "Laser printer", "24-inch monitor", "A4 printing paper", "Black toner cartridge",
               "Office chair", "Meeting table", "Annual maintenance service", "Accounting software subscription",
               "Network cable", "Wireless router", "External hard drive", "Keyboard", "Wireless mouse",
               "Filing cabinet", "Air conditioner", "Installation service", "Consulting hours",
               "Shipping and handling", "Security camera", "Projector"]
EN_CURRENCIES = ["USD", "GBP", "EUR", "SAR", "AED", "QAR"]
EN_PURPOSES = ["the supply and installation of computer equipment", "periodic maintenance of the premises",
               "the design and development of a resource management system", "transport and distribution services",
               "the supply of food products to the cafeteria", "interior finishing works for the offices"]
EN_CONTRACT_TITLES = ["Service Agreement", "Supply Agreement", "Maintenance Contract", "Works Contract",
                      "Cooperation Agreement"]
EN_CLAUSES = [
    "The preamble above forms an integral part of this Agreement and shall be read and construed with it.",
    "The Second Party shall perform the agreed works in accordance with the technical specifications attached hereto.",
    "This Agreement shall be valid for {months} months from the date of signature and may be renewed by written consent of both parties.",
    "The First Party shall pay the Second Party a total amount of {amount} {currency} in {parts} equal instalments.",
    "If the Second Party fails to deliver on time, the First Party may deduct a penalty of {pct}% for each week of delay.",
    "Both parties shall keep confidential all information and data disclosed to them under this Agreement.",
    "Neither party may assign this Agreement or any part thereof to a third party without the prior written consent of the other party.",
    "Either party may terminate this Agreement by giving {days} days' written notice if the other party breaches its obligations.",
    "This Agreement shall be governed by the laws of {country}, and the courts of {city} shall have jurisdiction over any dispute arising from it.",
    "The Second Party shall be liable for any damage caused to the property of the First Party during the performance of the works.",
    "All correspondence between the parties shall be sent to the addresses stated above or by email to {email}.",
    "The Second Party warrants the works performed for a period of {months} months from the date of final acceptance.",
    "This Agreement has been executed in two original copies, one for each party.",
]
EN_LETTER_SUBJECTS = ["Request for Quotation", "Notice of Change of Address", "Request for Contract Extension",
                      "Confirmation of Shipment Receipt", "Request for Payment", "Invitation to a Meeting"]
EN_LETTER_BODIES = [
    "We acknowledge receipt of your letter ref. {ref} dated {date}, which has been reviewed by the relevant department.",
    "Kindly provide us with a quotation for the supply of {product}, quantity {qty} units, including delivery time and payment terms.",
    "Please be informed that the meeting will be held on {weekday}, {date} at {hour}:00 a.m. at our head office.",
    "We would appreciate it if you could review the matter and take the necessary action as soon as possible.",
    "Please note that the outstanding amount is {amount} {currency} as per the attached invoice.",
    "Should you have any questions, please contact us on {phone} or by email at {email}.",
]
EN_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"]
EN_FORMS = [
    ("Employment Application Form", ["Joining the team", "Gaining new professional experience"]),
    ("Account Opening Form", ["Opening a current account", "Opening a savings account"]),
    ("Trainee Registration Form", ["Enrolling in the training programme", "Developing management skills"]),
    ("Leave Request Form", ["Annual leave", "Family matters", "Sick leave"]),
    ("Customer Information Update Form", ["Updating personal details", "Change of address and mobile number"]),
]

EMAIL_USERS = ["info", "sales", "contact", "admin", "accounts", "support", "hr", "procurement"]
EMAIL_DOMAINS = ["example.com", "company.sa", "mail.ae", "corp.qa", "business.com", "firm.co.uk"]


@dataclass
class Lang:
    code: str  # "ar" | "en" | "mixed"

    @property
    def rtl(self) -> bool:
        return self.code in ("ar", "mixed")


class Content:
    """Bộ sinh nội dung cho một tài liệu. `arabic_digits`: viết số bằng chữ số Ả Rập-Ấn."""

    def __init__(self, rng: random.Random, lang: str):
        self.rng = rng
        self.lang = lang
        self.ar = lang in ("ar", "mixed")
        self.arabic_digits = self.ar and rng.random() < 0.5

    # --- tiện ích ---
    def pick(self, seq):
        return self.rng.choice(seq)

    def num(self, s) -> str:
        s = str(s)
        return s.translate(AR_DIGITS) if self.arabic_digits else s

    def money(self, value: float) -> str:
        return self.num(f"{value:,.2f}")

    def date(self, years=(2022, 2026)) -> str:
        y, m, d = self.rng.randint(*years), self.rng.randint(1, 12), self.rng.randint(1, 28)
        fmt = self.pick(["{y}/{m:02d}/{d:02d}", "{d:02d}/{m:02d}/{y}"] if self.ar else
                        ["{d:02d}/{m:02d}/{y}", "{y}-{m:02d}-{d:02d}", "{d} {mon} {y}"])
        mon = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"][m - 1]
        return self.num(fmt.format(y=y, m=m, d=d, mon=mon))

    def phone(self) -> str:
        return self.num(self.pick(["+966 5{a} {b} {c}", "+971 5{a} {b} {c}", "+974 {b} {c}", "+44 20 {b} {c}"]).format(
            a=self.rng.randint(0, 9), b=self.rng.randint(100, 999), c=self.rng.randint(1000, 9999)))

    def email(self) -> str:
        return f"{self.pick(EMAIL_USERS)}@{self.pick(EMAIL_DOMAINS)}"

    def ref(self, prefix="") -> str:
        return f"{prefix}{self.rng.randint(1000, 99999)}"

    def code(self) -> str:
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        return "".join(self.rng.choice(letters) for _ in range(3)) + "-" + str(self.rng.randint(1000, 9999))

    # --- thực thể ---
    def person(self, male: bool = False) -> str:
        """male=True khi tên đi sau danh xưng nam (السيد)."""
        if self.ar:
            first = self.pick(AR_MALE if male else AR_MALE + AR_FEMALE)
            return f"{first} {self.pick(AR_MALE)} {self.pick(AR_FAMILY)}"
        return f"{self.pick(EN_FIRST)} {self.pick(EN_LAST)}"

    def company(self) -> str:
        if self.ar:
            name = f"{self.pick(AR_COMPANY_TYPES)} {self.pick(AR_COMPANIES)}"
            if self.lang == "mixed":
                name += f" ({self.pick(EN_COMPANIES)} {self.pick(EN_COMPANY_SUFFIX)})"
            return name
        return f"{self.pick(EN_COMPANIES)} {self.pick(EN_COMPANY_SUFFIX)}"

    def city_country(self):
        return self.pick(AR_CITIES if self.ar else EN_CITIES)

    def address(self) -> str:
        city, country = self.city_country()
        if self.ar:
            return f"{self.pick(AR_STREETS)}، {self.pick(AR_DISTRICTS)}، {city}، {country}"
        return f"{self.num(self.rng.randint(1, 250))} {self.pick(EN_STREETS)}, {city}, {country}"

    def job_title(self) -> str:
        return self.pick(AR_TITLES_JOB if self.ar else EN_TITLES_JOB)

    def currency(self) -> str:
        return self.pick(AR_CURRENCIES if self.ar else EN_CURRENCIES)

    def product(self) -> str:
        if self.lang == "mixed" and self.rng.random() < 0.6:
            return self.pick(MIXED_PRODUCTS)
        return self.pick(AR_PRODUCTS if self.ar else EN_PRODUCTS)

    def slots(self) -> dict:
        city, country = self.city_country()
        return {
            "months": self.num(self.pick([6, 12, 18, 24, 36])),
            "amount": self.money(self.rng.randint(5, 900) * 1000),
            "currency": self.currency(),
            "parts": self.num(self.pick([2, 3, 4, 6])),
            "pct": self.num(self.pick([0.5, 1, 2, 5])),
            "days": self.num(self.pick([15, 30, 60, 90])),
            "country": country,
            "city": city,
            "email": self.email(),
            "ref": self.num(self.ref()),
            "date": self.date(),
            "product": self.product(),
            "qty": self.num(self.rng.randint(2, 500)),
            "weekday": self.pick(AR_WEEKDAYS if self.ar else EN_WEEKDAYS),
            "hour": self.num(self.pick([8, 9, 10, 11])),
            "phone": self.phone(),
        }

    def fill(self, template: str) -> str:
        return template.format(**self.slots())
