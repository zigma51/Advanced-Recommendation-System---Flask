$("form[name=signup_form").submit(function (e) {
  var $form = $(this);
  var $error = $form.find(".error");
  var data1 = $form.serialize();

  $.ajax({
    url: "/signup/user/",
    type: "POST",
    data: data1,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/index/";
    },
    error: function (resp) {
      console.log(resp);
      $error.text(resp.responseJSON.error).removeClass("error--hidden");
    },
  });

  e.preventDefault();
});

$("form[name=login_form").submit(function (e) {
  var $form = $(this);
  var $error = $form.find(".error");
  var data1 = $form.serialize();

  $.ajax({
    url: "/login/user/",
    type: "POST",
    data: data1,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/index/";
    },
    error: function (resp) {
      console.log(resp);
      $error.text(resp.responseJSON.error).removeClass("error--hidden");
    },
  });

  e.preventDefault();
});
