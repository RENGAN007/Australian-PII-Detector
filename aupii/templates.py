"""Level 2 - document templates.

Slot syntax:  {NAME}  {NAME:option}  {NAME#2}  {NAME:option#2}

* NAME is either an entity label from labels.py (gets annotated), or
  FIRST_NAME (annotated as PERSON), or an unlabelled distractor such as
  {INVOICE_NO}, {ORDER_ID} or {FAKE_TFN} (see synth.DISTRACTORS).
* :option passes a setting to the generator, e.g. {AU_PHONE:mobile},
  {AU_DRIVER_LICENCE:VIC}, {AU_MEDICARE:irn}, {PERSON:surname_first}.
* #2 means "a second person / a second value". Within one document the
  same slot always gets the same value, and {PERSON}, {FIRST_NAME} and
  {EMAIL} all describe the same person, so documents stay consistent.

`cue=True` means the identifiers are introduced by cue words ("TFN:",
"ABN", "Medicare No"). `cue=False` templates show bare numbers, so the
model can't get away with only learning the cue words.

Templates are the unit of the train/val/test split (see build_dataset.py),
so add new ones freely - more templates means less template overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    doc_type: str
    cue: bool
    text: str


T = Template

PII_TEMPLATES: tuple[Template, ...] = (
    # --- payslips ---------------------------------------------------------
    T("payslip_01", "payslip", True,
      "{COMPANY}\nPAYSLIP - Pay period ending {DATE}\n\n"
      "Employee: {PERSON}\nEmployee address: {AU_ADDRESS}\n"
      "Tax File Number: {AU_TFN}\nEmployer ABN: {AU_ABN}\n\n"
      "Gross pay: {AMOUNT}\nHours worked: {HOURS}\n"
      "Paid to BSB {AU_BSB}, account {AU_BANK_ACCOUNT}\nPayslip reference: {REF_NO}"),
    T("payslip_02", "payslip", False,
      "{PERSON} | {AU_TFN} | {AU_BSB} {AU_BANK_ACCOUNT} | Net {AMOUNT} | {DATE}"),

    # --- employee onboarding ---------------------------------------------
    T("onboard_01", "onboarding_form", True,
      "NEW EMPLOYEE DETAILS\nFull name: {PERSON}\nDate of birth: {DATE_OF_BIRTH}\n"
      "Residential address: {AU_ADDRESS}\nMobile: {AU_PHONE:mobile}\nEmail: {EMAIL}\n"
      "TFN: {AU_TFN}\nBank: BSB {AU_BSB} Acc {AU_BANK_ACCOUNT}\nStart date: {DATE}"),
    T("onboard_02", "email", False,
      "Hi {FIRST_NAME#2},\n\nI've filled in the onboarding form. My details are below "
      "in case the upload didn't work:\n{PERSON}\n{AU_ADDRESS}\n{AU_PHONE}\n{AU_TFN}\n\n"
      "Thanks,\n{FIRST_NAME}"),
    T("onboard_03", "onboarding_form", True,
      "Super fund nomination - member {PERSON}, born {DATE_OF_BIRTH}, tax file no. "
      "{AU_TFN}. Employer reference {REF_NO}."),

    # --- invoices and quotes ---------------------------------------------
    T("invoice_01", "invoice", True,
      "TAX INVOICE\n{COMPANY}\nABN {AU_ABN}\nACN {AU_ACN}\nPhone {AU_PHONE:landline}\n"
      "Email {EMAIL}\n\nInvoice no: {INVOICE_NO}\nDate: {DATE}\n"
      "Bill to: {PERSON#2}, {AU_ADDRESS#2}\n\nTotal (inc. GST): {AMOUNT}\n"
      "Pay to BSB {AU_BSB} Account {AU_BANK_ACCOUNT}"),
    T("invoice_02", "invoice", False,
      "{COMPANY} ({AU_ABN}) - Quote {INVOICE_NO} for {PERSON}, valid until {DATE}. "
      "Call {AU_PHONE} with any questions."),
    T("invoice_03", "email", True,
      "Hi {FIRST_NAME}, please find attached invoice {INVOICE_NO} for {AMOUNT}. "
      "Our Australian Business Number is {AU_ABN}. Payment details: BSB {AU_BSB}, "
      "account number {AU_BANK_ACCOUNT}. Regards, {PERSON#2}"),
    T("invoice_04", "invoice", True,
      "Sole trader: {PERSON}\nABN: {AU_ABN}\nInvoice #{INVOICE_NO}\n"
      "Customer order {ORDER_ID}\nAmount due {AMOUNT} by {DATE}"),

    # --- health / Medicare -----------------------------------------------
    T("medicare_01", "health_form", True,
      "PATIENT REGISTRATION\nName: {PERSON}\nDOB: {DATE_OF_BIRTH}\n"
      "Medicare No: {AU_MEDICARE}\nPhone: {AU_PHONE}\nAddress: {AU_ADDRESS}\n"
      "Emergency contact: {PERSON#2} {AU_PHONE#2}"),
    T("medicare_02", "sms", False,
      "Booking confirmed for {PERSON} on {DATE} at {TIME}. Please bring your card "
      "({AU_MEDICARE}) and arrive 10 minutes early."),
    T("medicare_03", "health_form", True,
      "Medicare claim: patient {PERSON}, card number {AU_MEDICARE:irn}, date of "
      "service {DATE}, amount {AMOUNT}, reference {REF_NO}."),
    T("medicare_04", "chat", True,
      "hey can u update my medicare its {AU_MEDICARE} and dob is {DATE_OF_BIRTH}"),

    # --- banking ------------------------------------------------------------
    T("bank_01", "email", True,
      "Subject: Payment received\n\nDear {PERSON},\n\nWe received {AMOUNT} into account "
      "{AU_BSB} {AU_BANK_ACCOUNT} on {DATE}. Transaction ID: {ORDER_ID}.\n\n"
      "If this wasn't you, call us on {AU_PHONE:landline}."),
    T("bank_02", "bank_transfer", True,
      "Transfer {AMOUNT} to {PERSON#2}\nBSB: {AU_BSB}\nAccount: {AU_BANK_ACCOUNT}\n"
      "Description: rent {DATE}\nReceipt: {REF_NO}"),
    T("bank_03", "chat", False,
      "Mate, can you send the money to {AU_BSB} / {AU_BANK_ACCOUNT}? Name's {PERSON}. Cheers"),

    # --- rental and identity --------------------------------------------
    T("rental_01", "rental_application", True,
      "RENTAL APPLICATION\nApplicant: {PERSON}\nCurrent address: {AU_ADDRESS}\n"
      "Mobile: {AU_PHONE:mobile}\nEmail: {EMAIL}\nLicence state: VIC\n"
      "Licence no: {AU_DRIVER_LICENCE:VIC}\nPassport: {AU_PASSPORT}\n"
      "Reference: {PERSON#2}, {AU_PHONE#2}"),
    T("rental_02", "email", True,
      "Hi, I'm {PERSON} and I'd like to apply for the apartment. My NSW licence number "
      "is {AU_DRIVER_LICENCE:NSW} and you can reach me on {AU_PHONE}. "
      "Currently living at {AU_ADDRESS}."),
    T("rental_03", "id_check", False,
      "ID check - {PERSON}, {AU_DRIVER_LICENCE:QLD} (QLD), {DATE_OF_BIRTH}"),
    T("rental_04", "id_check", True,
      "Passport number {AU_PASSPORT}, expiry {DATE}. Full name as on passport: "
      "{PERSON:surname_first}"),
    T("identity_01", "email", True,
      "Identity verification: please upload a photo of your passport ({AU_PASSPORT}) "
      "or driver licence ({AU_DRIVER_LICENCE}). Account holder: {PERSON}."),
    T("travel_01", "booking", True,
      "Booking for {PERSON:surname_first}\nPassport {AU_PASSPORT}\nDOB {DATE_OF_BIRTH}\n"
      "Contact {AU_PHONE} / {EMAIL}\nBooking ref {REF_NO}"),
    T("passport_chat_01", "chat", True,
      "my passport number is {AU_PASSPORT} not the one on file, can you fix it? "
      "- {FIRST_NAME}"),

    # --- customer support -------------------------------------------------
    T("support_01", "chat", True,
      "[{TIME#1}] Agent {FIRST_NAME#2}: Can I get your full name and date of birth to verify?\n"
      "[{TIME#1}] Customer: {PERSON}, {DATE_OF_BIRTH}\n"
      "[{TIME#2}] Agent {FIRST_NAME#2}: Thanks. And the best number to call you back?\n"
      "[{TIME#2}] Customer: {AU_PHONE}"),
    T("support_02", "ticket", True,
      "customer says the TFN on file ({FAKE_TFN}) has a typo, correct TFN is {AU_TFN}. "
      "ticket {REF_NO}"),
    T("support_03", "chat", False,
      "Order {ORDER_ID} hasn't arrived. Ship to {AU_ADDRESS} please, my email is {EMAIL}"),
    T("support_04", "sms", False,
      "pls call me back on {AU_PHONE:mobile} re my account, {PERSON}"),

    # --- tax letters ------------------------------------------------------
    T("tax_01", "letter", True,
      "Our reference: {REF_NO}\n\n{PERSON}\n{AU_ADDRESS:multi_line}\n\nDear {FIRST_NAME},\n\n"
      "Your tax file number {AU_TFN} has been linked to the business with ABN {AU_ABN}. "
      "If you have questions, contact your tax agent."),
    T("tax_02", "letter", True,
      "Notice of assessment for {PERSON} (TFN {AU_TFN}) for the year ended 30 June. "
      "Amount payable: {AMOUNT}. Due {DATE}."),

    # --- companies ------------------------------------------------------------
    T("company_01", "company_extract", True,
      "Company extract: {COMPANY}, ACN {AU_ACN}, ABN {AU_ABN}. Director: {PERSON}, "
      "born {DATE_OF_BIRTH}, residing at {AU_ADDRESS}."),
    T("company_02", "company_extract", False,
      "Registered: {COMPANY} - {AU_ACN} - {AU_ADDRESS}"),
    T("supplier_01", "onboarding_form", True,
      "Supplier onboarding\nCompany: {COMPANY}\nAustralian Company Number: {AU_ACN}\n"
      "Contact person: {PERSON}\nPhone: {AU_PHONE:landline}"),

    # --- SMS / personal ---------------------------------------------------
    T("sms_01", "sms", False,
      "Hi {FIRST_NAME}, your parcel {TRACKING} arrives {DATE}. Reply STOP to opt out."),
    T("sms_02", "sms", False,
      "{FIRST_NAME}, it's {FIRST_NAME#2}. New number - {AU_PHONE:mobile}. Save it!"),
    T("cv_01", "resume", False,
      "{PERSON}\n{AU_PHONE:mobile} | {EMAIL} | {AU_ADDRESS}\n\nPROFILE\n"
      "Motivated professional with experience at {COMPANY}."),

    # --- bare lists (no cue words at all) ----------------------------------
    T("list_01", "list", False,
      "{PERSON}\n{DATE_OF_BIRTH}\n{AU_MEDICARE}\n{AU_PHONE}"),
    T("list_02", "list", False,
      "{PERSON}, {AU_TFN}, {AU_ABN}"),
)


# Documents with NO labelled PII. Several deliberately mention TFN / ABN /
# Medicare, or contain checksum-failing lookalikes, so the model learns
# that cue words or digit runs alone are not enough.
NEGATIVE_TEMPLATES: tuple[Template, ...] = (
    T("neg_01", "invoice", False,
      "Invoice {INVOICE_NO} from {COMPANY}: {AMOUNT} due {DATE}. Order ref {ORDER_ID}."),
    T("neg_02", "sms", False,
      "Your order {ORDER_ID} has shipped. Tracking number {TRACKING}. "
      "Estimated delivery {DATE}."),
    T("neg_03", "email", True,
      "My ABN is still pending, I'll send it through once it's approved."),
    T("neg_04", "notice", True,
      "Reminder: never share your TFN or Medicare number by email. "
      "Reference {FAKE_TFN} is a case number, not a tax file number."),
    T("neg_05", "email", False,
      "Meeting moved to {DATE} at {TIME}. Dial-in code {REF_NO}."),
    T("neg_06", "receipt", False,
      "Receipt {REF_NO}\n{COMPANY}\nTotal {AMOUNT}\nCard ending {CARD_LAST4}"),
    T("neg_07", "statement", False,
      "Customer number {CUSTOMER_NO}, account balance {AMOUNT} as at {DATE}."),
    T("neg_08", "ticket", True,
      "Note: ABN {FAKE_ABN} failed validation - please re-check with the supplier."),
    T("neg_09", "health_form", True,
      "The Medicare number entered ({FAKE_MEDICARE}) is invalid. Please try again."),
    T("neg_10", "email", True,
      "Please provide your tax file number on the attached form. "
      "Do not include it in this email."),
)

ALL_TEMPLATES: tuple[Template, ...] = PII_TEMPLATES + NEGATIVE_TEMPLATES
