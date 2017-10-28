tinyMCE.init({
  selector : "#checktext",
  plugins : "AtD,paste",
  paste_text_sticky : true,
  setup : function(ed) {
    ed.onInit.add(function(ed) {
      ed.pasteAsPlainText = true;
    });
    ed.onKeyUp.add(function(ed, l) {
      const editor_wordfiller = tinymce.get('checktext_wordfiller');
      editor_wordfiller.setContent(ed.getContent());
    });
  },  
  languagetool_i18n_no_errors : {
      "de-DE": "Keine Fehler gefunden."
  },
  languagetool_i18n_explain : {
      "de-DE": "Mehr Informationen..."
  },
  languagetool_i18n_ignore_once : {
      "de-DE": "Hier ignorieren"
  },
  languagetool_i18n_ignore_all : {
      "de-DE": "Fehler dieses Typs ignorieren"
  },
  languagetool_i18n_rule_implementation : {
      "de-DE": "Implementierung der Regel"
  },

  languagetool_i18n_current_lang :
      function() { return document.checkform.lang.value; },
  languagetool_rpc_url                 : "https://languagetool.org/api/v2/check",
  /* edit this file to customize how LanguageTool shows errors: */
  languagetool_css_url :
      "https://www.languagetool.org/online-check/" +
      "tiny_mce/plugins/atd-tinymce/css/content.css",
  theme                              : "advanced",
  theme_advanced_buttons1            : "",
  theme_advanced_buttons2            : "",
  theme_advanced_buttons3            : "",
  theme_advanced_toolbar_location    : "none",
  theme_advanced_toolbar_align       : "left",
  theme_advanced_statusbar_location  : "bottom",
  theme_advanced_path                : false,
  theme_advanced_resizing            : true,
  theme_advanced_resizing_use_cookie : false,
  gecko_spellcheck                   : false
});


tinyMCE.init({
  selector : "#checktext_wordfiller",
  plugins : "AtD,paste",
  paste_text_sticky : true,
  setup : function(ed) {
      ed.onInit.add(function(ed) {
          ed.pasteAsPlainText = true;
      });
    ed.onKeyUp.add(function(ed, l) {
      const editor_lt = tinymce.get('checktext');
      editor_lt.setContent(ed.getContent());
    });
  },  
  languagetool_i18n_no_errors : {
      "de-DE": "Keine Fehler gefunden."
  },
  languagetool_i18n_explain : {
      "de-DE": "Mehr Informationen..."
  },
  languagetool_i18n_ignore_once : {
      "de-DE": "Hier ignorieren"
  },
  languagetool_i18n_ignore_all : {
      "de-DE": "Fehler dieses Typs ignorieren"
  },
  languagetool_i18n_rule_implementation : {
      "de-DE": "Implementierung der Regel"
  },

  languagetool_i18n_current_lang : function() {
    return document.checkform.lang.value;
  },
  languagetool_rpc_url                 : "http://localhost:8081/v2/check",
  languagetool_css_url :
      "https://www.languagetool.org/online-check/" +
      "tiny_mce/plugins/atd-tinymce/css/content.css",
  theme                              : "advanced",
  theme_advanced_buttons1            : "",
  theme_advanced_buttons2            : "",
  theme_advanced_buttons3            : "",
  theme_advanced_toolbar_location    : "none",
  theme_advanced_toolbar_align       : "left",
  theme_advanced_statusbar_location  : "bottom",
  theme_advanced_path                : false,
  theme_advanced_resizing            : true,
  theme_advanced_resizing_use_cookie : false,
  gecko_spellcheck                   : false
});

function doit() {
  const langCode = document.checkform.lang.value;
  // if one of them returns before the other one has started checking, the first one might mix up the input text
  setTimeout(() => tinymce.get('checktext_wordfiller').execCommand("mceWritingImprovementTool", langCode), 250);
  tinymce.get('checktext').execCommand("mceWritingImprovementTool", langCode);
}

const exampleTexts = {
  "de-DE": "Ich glaube, das der Spieleabend gut besucht sein wir, da wir fiel Werbung gemacht haben. Was machst du den da? Ab wann seit ihr in der Uni? Ich bin gerade ihm Copyshop.",
  "en-US": "I didn’t no the answer, but he person told me the correct answer. We want too go to the museum no, but Peter isn’t here yet. Sara past the test yesterday. I lent him same money. Please turn of your phones.",
  "pt-PT": "Isso junta quanto? Sem dívida, algo inesperado ocorreu. Posso por a mesa? Filipe quer ganhar missa muscular. Eles são parceiros de lança há mais de 60 anos. Chorei a norte toda. Falei pôr telefone. [from tatoeba.org]"
}

function updateExampleText() {
  const editor_wordfiller = tinymce.get('checktext_wordfiller');
  const editor_lt = tinymce.get('checktext');
  chosenLanguage = document.getElementById("lang").value;
  editor_wordfiller.setContent(exampleTexts[chosenLanguage]);
  editor_lt.setContent(exampleTexts[chosenLanguage]);
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("lang").onchange = updateExampleText;
},false);
